-- {-# LANGUAGE AllowAmbiguousTypes                      #-}
{-# LANGUAGE BangPatterns                             #-}
{-# LANGUAGE DataKinds                                #-}
{-# LANGUAGE DeriveGeneric                            #-}
{-# LANGUAGE FlexibleContexts                         #-}
{-# LANGUAGE FlexibleInstances                        #-}
{-# LANGUAGE GADTs                                    #-}
{-# LANGUAGE LambdaCase                               #-}
{-# LANGUAGE PartialTypeSignatures                    #-}
{-# LANGUAGE PatternSynonyms                          #-}
{-# LANGUAGE RankNTypes                               #-}
{-# LANGUAGE ScopedTypeVariables                      #-}
{-# LANGUAGE TupleSections                            #-}
{-# LANGUAGE TypeApplications                         #-}
{-# LANGUAGE TypeOperators                            #-}
{-# LANGUAGE ViewPatterns                             #-}
{-# OPTIONS_GHC -Wno-incomplete-patterns              #-}
{-# OPTIONS_GHC -Wno-orphans                          #-}
{-# OPTIONS_GHC -Wno-unused-top-binds                 #-}
{-# OPTIONS_GHC -fno-warn-orphans                     #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures     #-}
{-# OPTIONS_GHC -fwarn-redundant-constraints          #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Extra.Solver    #-}
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise       #-}

module Main where
import           Control.DeepSeq
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Maybe
import           Control.Monad.Trans.State
import           Data.Bitraversable
import           Data.Foldable
import           Data.IDX
import           Data.List.Split
import           Data.Maybe
import           Data.Proxy
import           Data.Time.Clock
import           Data.Traversable
import           Data.Tuple
import           GHC.Generics                        (Generic)
import           GHC.TypeLits
import           GHC.TypeLits.Extra
import           Lens.Micro                          hiding ((&))
import           Mnist.Internal.Convolution
import           Numeric.Backprop                    hiding ((:&))
import           Numeric.LinearAlgebra.Static.Backprop                          -- hmatrix-backprop
import           Numeric.LinearAlgebra.Static.Vector (vecL, vecR, lVec, rVec)
import           Numeric.OneLiner
import           Text.Printf
import qualified Data.Vector                         as V
import qualified Data.Vector.Generic                 as VG
import qualified Data.Vector.Storable.Sized          as SVS
import qualified Data.Vector.Unboxed                 as VU
import qualified Numeric.LinearAlgebra               as HM  -- non-backprop hmatrix
import qualified Numeric.LinearAlgebra.Data          as HMD
import qualified Numeric.LinearAlgebra.Static        as HMS -- hmatrix with type-checked operations :D
import qualified System.Random.MWC                   as MWC
import qualified System.Random.MWC.Distributions     as MWC
import Debug.Trace

type Model p a b = forall z. Reifies z W
  => BVar z p
  -> BVar z a
  -> BVar z b

-- Custom data type for a tuple with strict values
data a :& b = !a :& !b
  deriving (Show, Generic)
infixr 2 :&

-- Do we actually need this? BVar already defines a similar instance
instance (NFData a, NFData b) => NFData (a :& b)
  where rnf (a :& b) = force a `seq` force b `seq` ()

instance (Num a, Num b) => Num (a :& b) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Fractional a, Fractional b) => Fractional (a :& b) where
    (/) = gDivide
    recip = gRecip
    fromRational = gFromRational

instance (Backprop a, Backprop b) => Backprop (a :& b)

instance (MWC.Variate a, MWC.Variate b, Num a, Num b) => MWC.Variate (a :& b) where
    uniform g = (:&) <$> MWC.uniform g <*> MWC.uniform g
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat n => MWC.Variate (R n) where
    uniform g = HMS.randomVector <$> MWC.uniform g <*> pure HMS.Uniform
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance (KnownNat m, KnownNat n) => MWC.Variate (L m n) where
    uniform g = HMS.uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

-- So if we have a BVar z (a :& b) (a BVar containing a tuple), then matching on (x :&& y) will give us x :: BVar z a and y :: BVar z b.
pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}

-- Just some lenses to help us get BVars out of the tuple
t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x
{-# INLINE t1 #-}

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y
{-# INLINE t2 #-}

feedForward
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForward (w :&& b) x = w #> x + b
{-# INLINE feedForward #-}

convolution
  :: ( KnownNat kernelSize
     , KnownNat filters
     , KnownNat stride
     , KnownNat inputRows
     , KnownNat inputCols
     , KnownNat (kernelSize * kernelSize * channels)
     , KnownNat (inputRows * channels)
     , KnownNat (((Div (inputRows - filters) stride) + 1) * filters)
     , KnownNat (((Div (inputCols - filters) stride) + 1))
     )
     => Proxy kernelSize
     -> Proxy filters
     -> Proxy stride
     -> Proxy inputRows
     -> Proxy inputCols
     -> Proxy channels
     -> Model
        (L (kernelSize * kernelSize * channels) filters)
        (L (inputRows * channels) inputCols)
        (L (((Div (inputRows - filters) stride) + 1) * filters) ((Div (inputCols - filters) stride) + 1))
convolution k' fs' st' ix' iy' cs' =
  liftOp2 . op2 $ \weights input ->
    (
      let ix = fromIntegral $ natVal ix'
          iy = fromIntegral $ natVal iy'
          kx = fromIntegral $ natVal k'
          ky = fromIntegral $ natVal k'
          sx = fromIntegral $ natVal st'
          sy = fromIntegral $ natVal st'
          ox = ((ix - (fromIntegral $ natVal fs')) `div` sx) + 1
          oy = ((iy - (fromIntegral $ natVal fs')) `div` sx) + 1
          ex = HMS.extract input
          ek = HMS.extract weights

          c  = vid2col kx ky sx sy ix iy ex
          mt = c HM.<> ek
          r  = col2vid 1 1 1 1 ox oy mt
          result = fromJust . HMS.create $ r
      in
          trace "convolution forwards" result

    , \dzdy ->
        let ix = fromIntegral $ natVal ix'
            iy = fromIntegral $ natVal iy'
            kx = fromIntegral $ natVal k'
            ky = fromIntegral $ natVal k'
            sx = fromIntegral $ natVal st'
            sy = fromIntegral $ natVal st'
            ox = ((ix - (fromIntegral $ natVal fs')) `div` sx) + 1
            oy = ((iy - (fromIntegral $ natVal fs')) `div` sx) + 1

            ex = HMS.extract input
            ek = HMS.extract weights
            eo = HMS.extract dzdy

            -- what is this actually doing?
            -- should be taking output from forward pass and reshaping it
            -- so we can do convolutions via matrix mult
            vs = vid2col 1 1 1 1 ox oy eo

            -- TODO: Gradient for weights -- I cannot get this to work -- WHY?
            -- It currently works if filters ^ 2 == weights rows (kernelSize ^ 2 * channels) -- but that's dumb
            c  = vid2col kx ky sx sy ix iy ex -- turn input image into columns
            dW = HM.tr c HM.<> vs

            -- Gradient for input -- This seems to work fine
            dX' = vs HM.<> HM.tr ek -- convolve (via matrix mult) output with transposed weights matrix
            dX = col2vid kx ky sx sy ix iy dX' -- stretch columns back into image dimensions
        in
            trace ("backwards") (fromJust . HMS.create $ dW, fromJust . HMS.create $ dX)
    )
{-# INLINE convolution #-}

flattenLayer :: (KnownNat o, KnownNat i, KnownNat (o * i), Reifies s W) => BVar s (L o i) -> BVar s (R (o * i))
flattenLayer = liftOp1 . op1 $ \input ->
  (
    let ex = HMS.extract input
        flattened = HMD.flatten ex
        result = fromJust . HMS.create $ flattened
      in
        trace "flatten forwards" result
  ,
    \dzdy ->
      let ex = HMS.extract dzdy
          ei = HMS.extract input
        in
          trace ("flatten backwards") fromJust . HMS.create $ HMD.reshape (HMD.cols ei) ex
  )
{-# INLINE flattenLayer #-}

-- TODO Add type annotation?
convLayer k fs st ix io cs p = vmap reLU . flattenLayer . convolution k fs st ix io cs p
{-# INLINE convLayer #-}

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))
{-# INLINE logistic #-}

feedForwardLog
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardLog wb = logistic . feedForward wb
{-# INLINE feedForwardLog #-}

reLU :: (Num a, Ord a) => a -> a
reLU x | x < 0     = 0
       | otherwise = x
{-# INLINE reLU #-}

feedForwardReLU
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardReLU wb = vmap reLU . feedForward wb
{-# INLINE feedForwardReLU #-}

softMax :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
softMax x = konst (1 / sumElements expx) * expx
   where
      expx = exp x
{-# INLINE softMax #-}

feedForwardSoftMax
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardSoftMax wb = softMax . feedForward wb
{-# INLINE feedForwardSoftMax #-}

crossEntropy
   :: (KnownNat n, Reifies s W)
   => R n
   -> BVar s (R n)
   -> BVar s Double
crossEntropy !targ !res = -(log res <.> constVar targ)
{-# INLINE crossEntropy #-}

netErr
   :: forall m n o p s. (KnownNat o, Reifies s W)
   => Model p (L m n) (R o)
   -> L m n
   -> R o
   -> BVar s p
   -> BVar s Double
netErr f !x !targ !p = crossEntropy targ $ f p $ auto x
{-# INLINE netErr #-}

trainModel
   :: forall m n o p. (KnownNat o, Backprop p, Fractional p)
   => Double                -- ^ learning rate
   -> Model p (L m n) (R o)
   -> p                     -- ^ initial params
   -> [(L m n, R o)]        -- ^ input and target pairs
   -> p                     -- ^ trained params
trainModel r f = foldl' (\p (x,y) -> p - realToFrac r * gradBP (netErr f x y) p)
{-# INLINE trainModel #-}

testNet
   :: forall m n o p. (KnownNat o)
   => Model p (L m n) (R o)
   -> p
   -> [(L m n, R o)]
   -> Double
testNet f !p !xs = sum (map (uncurry test) xs) / fromIntegral (length xs)
   where
      test :: L m n -> R o -> Double -- test if the max index is correct
      -- second argument here is using ViewPatterns extension of GHC
      test x (HMS.extract->t)
         | HM.maxIndex t == HM.maxIndex (HMS.extract r) = 1
         | otherwise                                    = 0
         where
            r :: R o
            r = evalBP2 f p x

-- Given two Models, we can define the composition of them both
-- as long as their input/output shapes match
(<~)
    :: (Backprop p, Backprop q)
    => Model  p       b c
    -> Model       q  a b
    -> Model (p :& q) a c
(f <~ g) (p :&& q) = f p . g q
infixr 8 <~
{-# INLINE (<~) #-}

-- The type wildcard here means GHC should infer the type of the model params
-- We could define it but then we'd have to update it manually every time we add a new "layer" here
-- Each layer's parameters (weights + bias) are tupled together so we can pattern match on them (see <~ operator above)
model :: Model _ (L 28 28) (R 10)
model =
   feedForwardSoftMax @100 @10
   <~ feedForwardLog @500 @100
   <~ feedForwardLog @2880 @500
   <~ convLayer (Proxy @5) (Proxy @5) (Proxy @1) (Proxy @28) (Proxy @28) (Proxy @1)
{-# INLINE model #-}

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
   Just train <- loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
   Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
   putStrLn "Loaded data."

   p0 <- MWC.uniformR (-0.5, 0.5) g
   -- print (show $ take 10 train)

   flip evalStateT p0 . forM_ [1..] $ \e -> do
      train' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList train) g
      liftIO $ printf "[Epoch %d]\n" (e :: Int)

      forM_ ([1..] `zip` chunksOf batch train') $ \(b, chnk) -> StateT $ \ps0 -> do
         printf "(Batch %d)\n" (b :: Int)

         t0 <- getCurrentTime
         newP <- evaluate . force $ trainModel (rate ^ fromIntegral e) model ps0 chnk
         t1 <- getCurrentTime
         printf "Trained on %d points in %s.\n" batch (show (t1 `diffUTCTime` t0))

         let trainScore = testNet model newP chnk
             testScore  = testNet model newP test
         printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
         printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)

         -- TODO: Serialize trained params so we can save/load them

         -- Because we are in the StateT monad, the next iteration of the loop
         -- will be passed newP (the updated parameters for the model)
         return ((), newP)
   where
      rate = 0.02
      batch = 500

loadMNIST
   :: FilePath
   -> FilePath
   -> IO (Maybe [(L 28 28, R 10)])
loadMNIST fpI fpL = runMaybeT $ do
   i <- MaybeT          $ decodeIDXFile       fpI
   l <- MaybeT          $ decodeIDXLabelsFile fpL
   d <- MaybeT . return $ labeledIntData l i
   r <- MaybeT . return $ for d (bitraverse mkImage mkLabel . swap)
   liftIO . evaluate $ force r
      where
         mkImage :: VU.Vector Int -> Maybe (L 28 28)
         mkImage = HMS.create . HMD.reshape 28 . VG.convert . VG.map (\i -> fromIntegral i / 255)
         mkLabel :: Int -> Maybe (R 10)
         mkLabel n = HMS.create $ HM.build 10 (\i -> if round i == n then 1 else 0)
