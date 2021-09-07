{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}

module Lib ( NetSpec (..)
           , Net
           , net
           , trainLoop
           , validLoop
           , train
           , preprocessData
           , OP (..)
           , OPData (..)
           , getDataFromNC
           , shuffleData
           , splitData

           , xyData
           , transformData
           , scaleData
           , transformData'
           , scaleData'
           ) where

import Torch hiding (take, floor)

import Data.NetCDF
import Data.NetCDF.Vector
import qualified Data.Vector.Storable as SV
import Foreign.C

import Data.Ord (compare, comparing)
import Data.Function (on)
import Data.Maybe (mapMaybe)
import qualified Data.List as L
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.Text.Lazy as T
import qualified Data.Text.Lazy.IO as T

import GHC.Generics
import GHC.Exts (IsList (fromList))

import Pipes
import qualified Pipes.Prelude as P

import Lens.Micro

import System.Random.Shuffle hiding (shuffle)

import Control.Monad (when, forM)
import Control.Monad.Cont (runContT)
import Control.Monad.Random hiding (fromList, split)

import Lucid
import Lucid.Html5
import Graphics.Plotly
import Graphics.Plotly.Lucid

import Network.Wai
import Network.HTTP.Types (status200)
import Network.Wai.Handler.Warp (run)

------------------------------------------------------------------------------
-- NEURAL NETWORK
------------------------------------------------------------------------------

-- | Neural Network Architecture Specification Type
data NetSpec = NetSpec { numX :: Int , numY :: Int }
    deriving (Show, Eq)

-- | Neural Network Architecture Specification
instance Randomizable NetSpec Net where
    sample NetSpec {..} = Net <$> sample (LinearSpec numX   32)     -- Layer 0
                              <*> sample (LinearSpec 32     128)    -- Layer 1
                              <*> sample (LinearSpec 128    256)    -- Layer 2
                              <*> sample (LinearSpec 256    512)    -- Layer 3
                              <*> sample (LinearSpec 512    1024)   -- Layer 4
                              <*> sample (LinearSpec 1024   512)    -- Layer 5
                              <*> sample (LinearSpec 512    256)    -- Layer 6
                              <*> sample (LinearSpec 256    128)    -- Layer 7
                              <*> sample (LinearSpec 128    32)     -- Layer 8
                              <*> sample (LinearSpec 32     numY)   -- Layer 9

-- | Neural Network Architecture Definition Type
data Net = Net { l0 :: Linear
               , l1 :: Linear
               , l2 :: Linear
               , l3 :: Linear
               , l4 :: Linear
               , l5 :: Linear
               , l6 :: Linear
               , l7 :: Linear
               , l8 :: Linear
               , l9 :: Linear
               }
    deriving (Generic, Show, Parameterized)

-- | Neural Network Architecture Definition
net :: Net -> Tensor -> Tensor
net Net {..} = linear l9 . relu
             . linear l8 . relu
             . linear l7 . relu
             . linear l6 . relu
             . linear l5 . relu
             . linear l4 . relu
             . linear l3 . relu
             . linear l2 . relu
             . linear l1 . relu
             . linear l0

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

-- | Training Loop
trainLoop :: Optimizer o => Net -> o -> Float -> ListT IO (Tensor, Tensor) 
          -> IO (Net, Float)
trainLoop model optim lr = P.foldM step begin done . enumerateData
    where step :: (Net, Float) -> ((Tensor, Tensor), Int) -> IO (Net, Float)
          step (m, l) ((x, y), i) = do
                let y' = net m x
                    l' = mseLoss y y'
                    l'' = l + (asValue l' :: Float)

                (m', _) <- runStep m optim l' (asTensor (lr :: Float))
                pure (m', l'')

          done = pure
          begin = pure (model, 0.0)

-- | Training Loop
validLoop :: Net -> ListT IO (Tensor, Tensor) -> IO (Net, Float)
validLoop model = P.foldM step begin done . enumerateData 
    where step :: (Net, Float) -> ((Tensor, Tensor), Int) -> IO (Net, Float)
          step (m, l) ((x, y), i) = do
                let y' = net m x
                    loss = asValue (l1Loss ReduceMean y y') :: Float
                pure (model, loss)

          done = pure
          begin = pure (model, 0.0)
                  
-- | Training
train :: [String] -> IO ()
train [] = return ()
train [[]] = return ()
train [_:_] = return ()
train (ncFileName:modelFileName:args) = do
    -- Loading Data from NetCDF
    rawData         <- getDataFromNC ncFileName (paramsX ++ paramsY)
    shuffledData    <- shuffleData rawData
    let sampledData = M.map (take numSamples) shuffledData

    -- Process data
    let (trainData, validData, scalerX, scalerY) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit sampledData
                             batchSize dev

    -- Turn data into torch dataset
    let numTrainBatches = length (inputs trainData)
        numValidBatches = length (inputs validData)
        trainSet = OP {dev = dev, numBatches = numTrainBatches, opData = trainData}
        validSet = OP {dev = dev, numBatches = numValidBatches, opData = validData}
    
    -- Neural Network Setup
    initModel' <- sample $ NetSpec numX numY

    let initModel = toDevice dev initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    (model, loss) <- foldLoop (initModel, []) numEpochs 
                    $ \(m, l) e -> do
                        let opts = datasetOpts numWorkers
                        (m', l') <- runContT (streamFromMap opts trainSet)
                                        $ trainLoop m optim learningRate . fst
                        putStrLn $ show e ++  " | Training Loss (MSE): " 
                                ++ show (l' / fromIntegral numTrainBatches)

                        -- (_, vl)  <- runContT (streamFromMap opts validSet)
                        --                 $ validLoop m' . fst
                        -- putStrLn $ "    +-> Validation Loss (MAE): " 
                        --         ++ show (vl / fromIntegral numValidBatches)

                        return (m', l':l)

    saveParams model modelFileName

    return ()

    where 
        -- ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
        -- modelFileName   = "./model.pt"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        maskX           = [0,1,0,0]
        maskY           = [1,0,1,0]
        numX            = length paramsX
        numY            = length paramsY
        numSamples      = 666666
        trainSplit      = 0.9
        lower           = 0
        upper           = 1
        numEpochs       = 100
        batchSize       = 2000
        numWorkers      = 25
        dev             = Device CUDA 1
        -- dev             = Device CPU 0
        learningRate    = 1.0e-3

------------------------------------------------------------------------------
-- INFERENCE
------------------------------------------------------------------------------

plotData :: IO ()
plotData = do
    rawData <- getDataFromNC ncFileName (paramsX ++ paramsY)

    let params = M.keys rawData
        Just vds = L.elemIndex "Vds" params
        Just vgs = L.elemIndex "Vgs" params
        Just vbs = L.elemIndex "Vbs" params
        Just gm  = L.elemIndex "gmoverid" params
        fd = (\d -> (roundn (d !! vds) 2 == 1.65) 
                 -- && (roundn (d !! vgs) 2 == 1.65) 
                 && (roundn (d !! vbs) 2 == 0.00))

    let traceData   = sortData gm . filterData fd $ rawData
        Just traceX = M.lookup "gmoverid" traceData
        Just traceY = M.lookup "idoverw" traceData
        trace       = line (aes & x .~ fst & y .~ snd) $ zip traceX traceY

    -- T.writeFile "plot.html" $ renderText $ doctypehtml_ $ do
    
    let htmlPlot = T.unpack . renderText . doctypehtml_ $ do
                        head_ $ do meta_ [charset_ "utf-8"] 
                                   plotlyCDN
                        body_ . toHtml $ plotly "myDiv" [trace]

    -- run 6666 (\_ res -> res $ responseLBS status200 [("Content-Type", "text/html")] htmlPlot)

    return ()
    where
        ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
        modelFileName   = "./model.pt"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        maskX           = [0,1,0,0]
        maskY           = [1,0,1,0]
        numX            = length paramsX
        numY            = length paramsY

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------

data OP = OP {dev :: Device, numBatches :: Int, opData :: OPData}

instance Dataset IO OP Int (Tensor, Tensor)
    where getItem OP {..} ix = pure ( toDevice dev . toDType Float 
                                                   . (!! ix) . inputs $ opData
                                    , toDevice dev . toDType Float 
                                                   . (!! ix) . outputs $ opData )
          keys OP {..} = S.fromList [ 0 .. (numBatches - 1) ]

data OPData = OPData { inputs :: [Tensor], outputs :: [Tensor] }

type SVRet a = IO (Either NcError (SV.Vector a))

-- | Reads data from a NetCDF file and returns it as a map
getDataFromNC :: String -> [String] -> IO (M.Map String [Float])
getDataFromNC fileName params = do
    Right ncFile <- openFile fileName
    vals <- forM params 
                 (\param -> do
                    let (Just var) = ncVar ncFile param
                    Right vec <- get ncFile var :: SVRet CFloat
                    let val = map realToFrac (SV.toList vec)
                    return val)
    return (M.fromList $ zip params vals)

-- | Round to n digits
roundn :: Float -> Int -> Float
roundn d n = fromInteger (round $ d * (10^n)) / (10.0^^n)

-- | Sort data based on n-th element
sortData :: Int -> M.Map String [Float] -> M.Map String [Float]
sortData n m = M.fromList . zip p . L.transpose . L.sortBy f 
             . L.transpose . mapMaybe (`M.lookup` m) $ p
    where p = M.keys m
          f = compare `on` (!! n)

-- | Filter a datamap
filterData :: ([Float] -> Bool) -> M.Map String [Float] -> M.Map String [Float]
filterData f m = M.fromList . zip p . L.transpose . L.filter f 
               . L.transpose . mapMaybe (`M.lookup` m) $ p
    where p = M.keys m

-- | Shuffles data in a assoc map
shuffleData :: M.Map String [Float] -> IO (M.Map String [Float])
shuffleData dataMap = zipup <$> shuff params 
    where params = M.keys dataMap
          zipup  = M.fromList . zip params . L.transpose
          shuff  = evalRandIO . shuffleM . L.transpose 
                 . mapMaybe (`M.lookup` dataMap)

-- | Splits data into two sets according to a ratio
splitData :: M.Map String [Float] -> Float 
          -> (M.Map String [Float], M.Map String [Float])
splitData m s = ( M.map (take numTrainSamples) m
                     , M.map (drop numTrainSamples) m )
    where (Just numSamples) = fmap length (M.lookup (head (M.keys m)) m)
          numTrainSamples   = floor . (* s) . fromIntegral $ numSamples

-- | Splits data into two sets of columns (x, y)
xyData :: M.Map String [Float] -> [String] -> [String] -> (Tensor, Tensor)
xyData m x y = ( toTensor (mapMaybe (`M.lookup` m) x :: [[Float]])
               , toTensor (mapMaybe (`M.lookup` m) y :: [[Float]]) )
    where toTensor = transpose (Dim 0) (Dim 1) . asTensor 

-- | Applys `ln|x|` to each feature x of the given tensor. Where `x` can be
-- | masked with an [Int].
transformData :: [Int] -> Tensor -> Tensor
transformData m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.log
                  . (+ asTensor (1 :: Float))
                  . Torch.abs 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor (m :: [Int])
          t' = t * (1 - m')

-- | Applys `e^x` to each column x of the given tensor. Where `x` can be masked
-- | with [Int].
transformData' :: [Int] -> Tensor -> Tensor
transformData' m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.exp
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor (m :: [Int])
          t' = t * (1 - m')

-- | Scale data such that x ∈ [a;b]. Returns a tuple, with the scaled Tensor
-- | and a Tensor [2,num-features] with min and max values per feature for
-- | unscaling.
scaleData :: Float -> Float -> Tensor -> (Tensor, Tensor)
scaleData a b x = ( ((x - xMin) * (b' - a')) / (xMax - xMin)
                  , Torch.cat (Dim 0) [ Torch.reshape [1,-1] xMin
                                      , Torch.reshape [1,-1] xMax ])
    where (xMin, _) = Torch.minDim (Dim 0) RemoveDim x
          (xMax, _) = Torch.maxDim (Dim 0) RemoveDim x
          a' = asTensor (a :: Float)
          b' = asTensor (b :: Float)
          
-- | Un-Scale data where x ∈ [a;b] for a given min and max.
scaleData' :: Float -> Float -> Tensor -> Tensor -> Tensor
scaleData' a b s y = ((y - a') / (b' - a') * (xMax - xMin)) + xMin
    where xMin = Torch.select 0 0 s
          xMax = Torch.select 0 1 s
          a' = asTensor (a :: Float)
          b' = asTensor (b :: Float)

-- | 'preprocessData' is a convenience function to clean up main.
-- | Arguments:
-- |    computing device :: Device
-- |    lower limit scaled :: FLoat
-- |    upper limit scaled :: FLoat
-- |    transformation mask X :: [Int]
-- |    transformation mask Y :: [Int]
-- |    parameters X :: [String]
-- |    parameters Y :: [String]
-- |    train test split ratio :: Float
-- |    NetCDF Data map as obtained by `getDataFromNC`
-- |    batch size :: Int
-- |    target computing device :: Device
-- | Returns a tuple  split into train and validation sets. Additionaly scalers
-- | are returned for unscaling the data later.
-- |    -> (trainingData, validationData, scalerX, scalerY)
preprocessData :: Float -> Float -> [Int] -> [Int] -> [String] -> [String] 
                -> Float -> M.Map String [Float] -> Int -> Device
                -> (OPData, OPData, Tensor, Tensor)
preprocessData lo hi mX mY pX pY splt dat bs dev 
        = ( trainData, validData, scalerX, scalerY)
    where (rawTrain, rawValid)       = splitData dat splt
          (rawTrainX, rawTrainY)     = xyData rawTrain pX pY
          (rawValidX, rawValidY)     = xyData rawValid pX pY
          trafoTrainX                = transformData mX rawTrainX
          trafoTrainY                = transformData mY rawTrainY
          trafoValidX                = transformData mX rawValidX
          trafoValidY                = transformData mY rawValidY
          (trainX, scalerX)          = scaleData lo hi trafoTrainX
          (trainY, scalerY)          = scaleData lo hi trafoTrainY
          (validX, _)                = scaleData lo hi trafoValidX
          (validY, _)                = scaleData lo hi trafoValidY
          trainData                  = OPData (split bs (Dim 0) . toDType Float 
                                                                $ trainX) 
                                              (split bs (Dim 0) . toDType Float 
                                                                $ trainY)
          validData                  = OPData (split bs (Dim 0) . toDType Float 
                                                                $ validX) 
                                              (split bs (Dim 0) . toDType Float 
                                                                $ validY)
