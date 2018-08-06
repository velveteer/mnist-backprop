{-# LANGUAGE BangPatterns                             #-}
{-# LANGUAGE DataKinds                                #-}
{-# LANGUAGE DeriveGeneric                            #-}
{-# LANGUAGE FlexibleContexts                         #-}
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
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise       #-}
{-# OPTIONS_GHC -fwarn-redundant-constraints          #-}

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
import           Data.Time.Clock
import           Data.Traversable
import           Data.Tuple
import           GHC.Generics                        (Generic)
import           GHC.TypeLits
import           Lens.Micro                          hiding ((&))
import           Numeric.Backprop (Backprop, (^^.), auto, isoVar2, evalBP, evalBP2, gradBP, constVar)
import           Numeric.LinearAlgebra.Static.Backprop -- hmatrix-backprop
import           Numeric.LinearAlgebra.Static.Vector (vecL, vecR, lVec, rVec)
import           Numeric.OneLiner
import           Text.Printf
import qualified Data.Vector                         as V
import qualified Data.Vector.Generic                 as VG
import qualified Data.Vector.Storable.Sized          as SVS
import qualified Data.Vector.Unboxed                 as VU
import qualified Numeric.LinearAlgebra               as HM -- non-backprop hmatrix
import qualified Numeric.LinearAlgebra.Static        as HMS -- hmatrix with type-checked operations :D
import qualified System.Random.MWC                   as MWC
import qualified System.Random.MWC.Distributions     as MWC

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

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))
{-# INLINE logistic #-}

feedForwardLog
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardLog wb = logistic . feedForward wb
{-# INLINE feedForwardLog #-}

relu :: Floating a => a -> a
relu x = absx - 0.5 * absx
   where absx = (abs x) + x
{-# INLINE relu #-}

feedForwardReLu
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardReLu wb = relu . feedForward wb
{-# INLINE feedForwardReLu #-}

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
   :: forall i o p s. (KnownNat o, Reifies s W)
   => Model p (R i) (R o)
   -> R i
   -> R o
   -> BVar s p
   -> BVar s Double
netErr f !x !targ !p = crossEntropy targ $ f p $ auto x
{-# INLINE netErr #-}

trainModel
   :: forall i o p. (KnownNat o, Backprop p, Fractional p)
   => Double                -- ^ learning rate
   -> Model p (R i) (R o)
   -> p                     -- ^ initial params
   -> [(R i, R o)]          -- ^ input and target pairs
   -> p                     -- ^ trained params
trainModel r f = foldl' (\p (x,y) -> p - realToFrac r * gradBP (netErr f x y) p)
{-# INLINE trainModel #-}

testNet
   :: forall i o p. (KnownNat o)
   => Model p (R i) (R o)
   -> p
   -> [(R i, R o)]
   -> Double
testNet f !p !xs = sum (map (uncurry test) xs) / fromIntegral (length xs)
   where
      test :: R i -> R o -> Double -- test if the max index is correct
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

-- The type wildcard here means GHC can infer the type of the model params
-- If you look up at the composition operator, you can see how the tupling
-- of model parameters can nest infinitely, and this would be tedious to write by hand
model :: Model _ (R 784) (R 10)
model =
   feedForwardSoftMax @100 @10
   <~ feedForwardReLu @300 @100
   <~ feedForwardReLu @784 @300
{-# INLINE model #-}

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
   Just train <- loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
   Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
   putStrLn "Loaded data."

   p0 <- MWC.uniformR (-0.5, 0.5) g
   flip evalStateT p0 . forM_ [1..] $ \e -> do
      train' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList train) g
      liftIO $ printf "[Epoch %d]\n" (e :: Int)

      forM_ ([1..] `zip` chunksOf batch train') $ \(b, chnk) -> StateT $ \ps0 -> do
         printf "(Batch %d)\n" (b :: Int)

         t0 <- getCurrentTime
         newP <- evaluate . force $ trainModel (rate * fromIntegral e) model ps0 chnk
         t1 <- getCurrentTime
         printf "Trained on %d points in %s.\n" batch (show (t1 `diffUTCTime` t0))

         let trainScore = testNet model newP chnk
             testScore  = testNet model newP test
         printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
         printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)

         -- Because we are in the StateT monad, the next iteration of the loop
         -- will be passed newP (the updated parameters for the model)
         return ((), newP)
   where
      rate = 0.001
      batch = 500

loadMNIST
   :: FilePath
   -> FilePath
   -> IO (Maybe [(R 784, R 10)])
loadMNIST fpI fpL = runMaybeT $ do
   i <- MaybeT          $ decodeIDXFile       fpI
   l <- MaybeT          $ decodeIDXLabelsFile fpL
   d <- MaybeT . return $ labeledIntData l i
   r <- MaybeT . return $ for d (bitraverse mkImage mkLabel . swap)
   liftIO . evaluate $ force r
      where
         mkImage :: VU.Vector Int -> Maybe (R 784)
         mkImage = HMS.create . VG.convert . VG.map (\i -> fromIntegral i / 255)
         mkLabel :: Int -> Maybe (R 10)
         mkLabel n = HMS.create $ HM.build 10 (\i -> if round i == n then 1 else 0)
