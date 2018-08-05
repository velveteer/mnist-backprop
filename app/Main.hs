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
import           Lens.Micro.TH
import           Numeric.Backprop (Backprop, (^^.), auto, isoVar2, evalBP, evalBP2, gradBP, constVar)
import           Numeric.LinearAlgebra.Static.Backprop -- hmatrix-backprop
import           Numeric.LinearAlgebra.Static.Vector (vecL, vecR, lVec, rVec)
import           Numeric.OneLiner
import           System.Random
import           Text.Printf
import qualified Data.Vector                         as V
import qualified Data.Vector.Generic                 as VG
import qualified Data.Vector.Storable.Sized          as SVS
import qualified Data.Vector.Unboxed                 as VU
import qualified Numeric.LinearAlgebra               as HM -- non-backprop hmatrix
import qualified Numeric.LinearAlgebra.Static        as HMS -- hmatrix with type-checked operations :D
import qualified System.Random.MWC                   as MWC
import qualified System.Random.MWC.Distributions     as MWC

-- -- L is from hmatrix (it represents a matrix) -- http://dis.um.es/~alberto/hmatrix/static.html
-- -- R is a regular old vector
-- data Layer i o =
--    Layer { _lWeights :: !(L o i)
--          , _lBiases  :: !(R o)
--          }
--    deriving (Show, Generic)
--    -- Small note: deriving tells the compiler to automatically generate instances for
--    -- these classes. The Generic type class means the type can be represented "generically",
--    -- as a combination of simpler product and sum types.
--    -- Generics let us get a lot of behaviors for free.

-- -- KnownNat class means the type params in Layer can be represented as natural numbers at the type level
-- -- but they can be turned into integers at run-time
-- -- This lets the type system check equality of numbers (because natural numbers can be defined inductively)
-- -- I think this is mainly used for when hmatrix checks the structural correctness of its computations

-- -- Let the compiler resolve a Backprop instance for our Layer type
-- -- https://backprop.jle.im/04-the-backprop-typeclass.html
-- -- This works because Layer has a Generic instance (so we do not need to implement the Backprop class functions)
-- instance (KnownNat i, KnownNat o) => Backprop (Layer i o)

-- -- NFData is from deepseq. It means every type in Layer can be evaluated strictly.
-- -- We want deepseq for fully evaluating the data structure, so that we don't
-- -- wind up with space leaks.
-- instance NFData (Layer i o)

-- -- This is a helper that uses Template Haskell, a form of metaprogramming (like macros in Scheme).
-- -- It generates lens functions for each type defined in the Layer product (using the type label as a guide)
-- -- So we should get some lenses called "lWeights" and "lBiases", which we can use as getters
-- makeLenses ''Layer

-- -- A Network data type is constructed by passing in the input layer, the hidden layers,
-- -- and the output layer.
-- data Network i h1 h2 h3 o =
--    Net { _nLayer1 :: !(Layer i  h1)
--        , _nLayer2 :: !(Layer h1 h2)
--        , _nLayer3 :: !(Layer h2 h3)
--        , _nLayer4 :: !(Layer h3 o)
--        }
--    deriving (Show, Generic)

-- instance (KnownNat i, KnownNat h1, KnownNat h2, KnownNat h3, KnownNat o) => Backprop (Network i h1 h2 h3 o)
-- instance NFData (Network i h1 h2 h3 o)
-- makeLenses ''Network

-- -- Num and Fractional instances. These rely on the Generic implementations of
-- -- the operations defined on the Num and Fractional classes.
-- -- Note in trainStep we subtract a fractional from the Network -- so the compiler needs to know
-- -- what to dispatch to for doing fractional subtraction with a Network
-- instance (KnownNat i, KnownNat o) => Num (Layer i o) where
--    (+)         = gPlus
--    (-)         = gMinus
--    (*)         = gTimes
--    negate      = gNegate
--    abs         = gAbs
--    signum      = gSignum
--    fromInteger = gFromInteger

-- instance ( KnownNat i
--          , KnownNat h1
--          , KnownNat h2
--          , KnownNat h3
--          , KnownNat o
--          ) => Num (Network i h1 h2 h3 o) where
--    (+)         = gPlus
--    (-)         = gMinus
--    (*)         = gTimes
--    negate      = gNegate
--    abs         = gAbs
--    signum      = gSignum
--    fromInteger = gFromInteger

-- instance (KnownNat i, KnownNat o) => Fractional (Layer i o) where
--    (/)          = gDivide
--    recip        = gRecip
--    fromRational = gFromRational

-- instance ( KnownNat i
--          , KnownNat h1
--          , KnownNat h2
--          , KnownNat h3
--          , KnownNat o
--          ) => Fractional (Network i h1 h2 h3 o) where
--    (/)          = gDivide
--    recip        = gRecip
--    fromRational = gFromRational

-- -- Now some actual math

-- runLayer
--    :: (KnownNat i, KnownNat o, Reifies s W)
--    => BVar s (Layer i o)
--    -> BVar s (R i)
--    -> BVar s (R o)
-- runLayer l x = (l ^^. lWeights) #> x + (l ^^. lBiases)
-- {-# INLINE runLayer #-}

-- -- Already backprop aware because BVar provides an instance for Floating
-- logistic :: Floating a => a -> a
-- logistic x = 1 / (1 + exp (-x))
-- {-# INLINE logistic #-}

-- batchNormalization :: Floating a => a -> a -> a -> a
-- batchNormalization x gamma beta = x

-- runNetwork
--    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat h3, KnownNat o, Reifies s W)
--    => BVar s (Network i h1 h2 h3 o)
--    -> R i
--    -> BVar s (R o)
-- runNetwork n = softMax
--             . runLayer (n ^^. nLayer4)
--             . logistic
--             . runLayer (n ^^. nLayer3)
--             . logistic
--             . runLayer (n ^^. nLayer2)
--             . logistic
--             . runLayer (n ^^. nLayer1)
--             . constVar
-- {-# INLINE runNetwork #-}

-- crossEntropy
--    :: (KnownNat n, Reifies s W)
--    => R n
--    -> BVar s (R n)
--    -> BVar s Double
-- crossEntropy targ res = -(log res <.> constVar targ)
-- {-# INLINE crossEntropy #-}

-- netErr
--    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat h3, KnownNat o, Reifies s W)
--    => R i
--    -> R o
--    -> BVar s (Network i h1 h2 h3 o)
--    -> BVar s Double
-- netErr x targ n = crossEntropy targ (runNetwork n x)
-- {-# INLINE netErr #-}

-- trainStep
--    :: forall i h1 h2 h3 o. (KnownNat i, KnownNat h1, KnownNat h2, KnownNat h3, KnownNat o)
--    => Double             -- ^ learning rate
--    -> R i                -- ^ input
--    -> R o                -- ^ target
--    -> Network i h1 h2 h3 o  -- ^ initial network
--    -> Network i h1 h2 h3 o
-- trainStep r !x !targ !n = n - realToFrac r * gradBP (netErr x targ) n
-- {-# INLINE trainStep #-}

-- trainList
--    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat h3, KnownNat o)
--    => Double             -- ^ learning rate
--    -> [(R i, R o)]       -- ^ input and target pairs
--    -> Network i h1 h2 h3 o  -- ^ initial network
--    -> Network i h1 h2 h3 o
-- trainList r = flip $ foldl' (\n (x,y) -> trainStep r x y n)
-- {-# INLINE trainList #-}

type Model p a b = forall z. Reifies z W
  => BVar z p
  -> BVar z a
  -> BVar z b

-- Custom data type for a tuple with strict values
data a :& b = !a :& !b
  deriving (Show, Generic)
infixr 2 :&

instance (NFData a, NFData b) => NFData (a :& b)

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

instance (Random a, Random b) => Random (a :& b) where
    random g0 = (x :& y, g2)
      where
        (x, g1) = random g0
        (y, g2) = random g1
    randomR (x0 :& y0, x1 :& y1) g0 = (x :& y, g2)
      where
        (x, g1) = randomR (x0, x1) g0
        (y, g2) = randomR (y0, y1) g1

instance (KnownNat n, KnownNat m) => Random (L n m) where
    random = runState . fmap vecL $ SVS.replicateM (state random)
    randomR (xs,ys) = runState . fmap vecL $ SVS.zipWithM (curry (state . randomR))
        (lVec xs) (lVec ys)

instance (KnownNat n) => Random (R n) where
    random = runState $ vecR <$> SVS.replicateM (state random)
    randomR (xs,ys) = runState . fmap vecR $ SVS.zipWithM (curry (state . randomR))
        (rVec xs) (rVec ys)

-- So if we have a BVar z (a :& b) (a BVar containing a tuple), then matching on (x :&& y) will give us x :: BVar z a and y :: BVar z b.
pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}

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
crossEntropy targ res = -(log res <.> constVar targ)
{-# INLINE crossEntropy #-}

netErr
   :: forall i o p s. (KnownNat i, KnownNat o, Reifies s W)
   => Model p (R i) (R o)
   -> R i
   -> R o
   -> BVar s p
   -> BVar s Double
netErr f !x !targ !p = crossEntropy targ $ f p $ auto x
{-# INLINE netErr #-}

trainModel
   :: forall i o p. (KnownNat i, KnownNat o, Backprop p, Fractional p)
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

(<~)
    :: (Backprop p, Backprop q)
    => Model  p       b c
    -> Model       q  a b
    -> Model (p :& q) a c
(f <~ g) (p :&& q) = f p . g q
infixr 8 <~
{-# INLINE (<~) #-}

model :: Model _ (R 784) (R 10)
model =
   feedForwardSoftMax @100 @10
   <~ feedForwardReLu @300 @100
   <~ feedForwardReLu @500 @300
   <~ feedForwardReLu @784 @500
{-# INLINE model #-}

initParams
    :: (Fractional p, Random p)
    => Model p (R i) (R o)      -- ^ model to train
    -> IO p                     -- ^ parameter guess
initParams m = do
    p0 <- (/ 10) . subtract 0.5 <$> randomIO
    return p0

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
   Just train <- loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
   Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
   putStrLn "Loaded data."

   p0 <- initParams model

   flip evalStateT p0 . forM_ [1..] $ \e -> do
      train' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList train) g
      liftIO $ printf "[Epoch %d]\n" (e :: Int)

      forM_ ([1..] `zip` chunksOf batch train') $ \(b, chnk) -> StateT $ \ps0 -> do
         printf "(Batch %d)\n" (b :: Int)

         t0 <- getCurrentTime
         newP <- evaluate . force $ trainModel 0.01 model ps0 chnk
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

-- instance KnownNat n => MWC.Variate (R n) where
--    uniform g = HMS.randomVector <$> MWC.uniform g <*> pure HMS.Uniform
--    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

-- instance (KnownNat m, KnownNat n) => MWC.Variate (L m n) where
--    uniform g = HMS.uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
--    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

-- instance (KnownNat i, KnownNat o) => MWC.Variate (Layer i o) where
--    uniform g = Layer <$> MWC.uniform g <*> MWC.uniform g
--    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

-- instance ( KnownNat i
--          , KnownNat h1
--          , KnownNat h2
--          , KnownNat h3
--          , KnownNat o
--          )
--       => MWC.Variate (Network i h1 h2 h3 o) where
--    uniform g = Net <$> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g
--    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g
