{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Lib ( NetSpec (..)
           , Net
           , net
           , toDoubleGPU
           , toFloatCPU
           , trainNet
           , trainNet'
           , getDataFromNC
           , shuffleData
           , preprocessData
           , preprocessData'
           ) where

import Pipes
import qualified Pipes.Prelude as P
import Torch hiding (take, floor)
import Torch.Lens
import Data.NetCDF
import Data.NetCDF.Vector
import Data.Function (on)
import Data.Maybe
import Data.List.Split hiding (split)
import Foreign.C
import GHC.Generics
import System.Random.Shuffle
import Control.Monad.Random hiding (fromList, split)
import Control.Monad.Cont (runContT)
import qualified Data.Map as M
import qualified Data.Set as S
import qualified Data.List as L
import qualified Data.Vector.Storable as SV
import Graphics.Vega.VegaLite hiding (sample, shape)

-- | Main Computing Device
computingDevice :: Device
computingDevice = Device CUDA 1

------------------------------------------------------------------------------
-- NEURAL NETWORK
------------------------------------------------------------------------------

-- | Neural Network Specification
data NetSpec = NetSpec { numX :: Int , numY :: Int }
    deriving (Show, Eq)

-- | Network Architecture
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
    deriving (Generic, Show, Torch.Parameterized)

-- | Neural Network Weight initialization
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

-- | Neural Network Forward Pass
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
             . linear l0 . relu

-- | Convert model to Double on GPU
toDoubleGPU :: forall a. HasTypes a Tensor => a -> a
toDoubleGPU = over (types @ Tensor @a) (toDevice computingDevice . toType Double)

-- | Convert model to Float on CPU
toFloatCPU :: forall a. HasTypes a Tensor => a -> a
toFloatCPU = over (types @ Tensor @a) (toDevice computingDevice . toType Float)

------------------------------------------------------------------------------
-- TRAINING
------------------------------------------------------------------------------

-- | Training Loop
trainLoop :: Optimizer o => Net -> o -> Tensor -> ListT IO (Tensor, Tensor) 
          -> IO (Net, Double)
trainLoop model optim lr = P.foldM step begin done . enumerateData
    where step :: (Net, Double) -> ((Tensor, Tensor), Int) -> IO (Net, Double)
          -- step (m, l) ((x, y), i) = do
          step (m, l) ((x, y), _) = do
                let y' = net m x
                    l' = mseLoss y y'
                    l'' = l + (asValue l' :: Double)

                (m', _) <- runStep m optim l' lr
                pure (m', l'')

          done = pure
          begin = pure (model, 0.0)

-- | Validation Loop
validLoop :: Net -> ListT IO (Tensor, Tensor) -> IO (Net, Double)
validLoop model = P.foldM step begin done . enumerateData 
    where step :: (Net, Double) -> ((Tensor, Tensor), Int) -> IO (Net, Double)
          step (m, l) ((x, y), _) = do
                let y' = net m x
                    loss = asValue (l1Loss ReduceMean y y') :: Double
                pure (model, loss)

          done = pure
          begin = pure (model, 0.0)
                  
-- | Training with Datasets/Streams
trainNet :: OPData -> OPData -> IO Net
trainNet trainData validData = do
    -- Neural Network Setup
    initModel <- toDoubleGPU <$> sample (NetSpec numX numY)

    -- Optimizer
    let optim = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    model' <- foldLoop initModel numEpochs $ \m e -> do
        -- Training
        (m', l') <- runContT (streamFromMap opts trainSet)
                        $ trainLoop m optim learningRate . fst

        putStrLn $ show e ++  " | Training Loss (MSE): " 
                ++ show (l' / fromIntegral numTrainBatches)

        -- Validation
        (m'', l'') <- runContT (streamFromMap opts trainSet)
                        $ validLoop m . fst

        putStrLn $ show e ++  " | Validation Loss (MAE): " 
                ++ show (l'' / fromIntegral numValidBatches)

        return m'
    
    -- Save final model
    save (toDependent <$> flattenParameters model') ptFile
    saveParams (flattenParameters model') ptFile

    return model'

    where opts            = datasetOpts 25
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          numX            = length paramsX
          numY            = length paramsY
          numEpochs       = 42
          learningRate    = asTensor' (1.0e-3 :: Double) ( withDType Double 
                                                         . withDevice computingDevice 
                                                         $ defaultOpts )
          numTrainBatches = length (inputs trainData)
          trainSet        = OP { opdev = computingDevice
                               , numBatches = numTrainBatches
                               , opData = trainData }
          numValidBatches = length (inputs validData)
          validSet        = OP { opdev = computingDevice
                               , numBatches = numValidBatches
                               , opData = validData }
          ptFile          = "./models/prehsept/model.pt"

-- | Training with raw Tensors, no streams/datasets
trainNet' :: (Tensor, Tensor) -> (Tensor, Tensor) -> IO Net
trainNet' (trainX, trainY) (validX, validY) = do
    -- Neural Network Setup
    initModel <- toDoubleGPU <$> sample (NetSpec numX numY)
    let optim = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    let numTrainSamples = head (shape trainX)
        trainIdx        = chunksOf batchSize [0 .. (numTrainSamples - 1)]
        trainIters      = length trainIdx - 1
        numValidSamples = head (shape validX)
        validIdx        = chunksOf batchSize [0 .. (numValidSamples - 1)]
        validIters      = length validIdx - 1

    -- Iterate over all Epochs
    trainedModel <- foldLoop initModel numEpochs $ \model epoch -> do
        -- Iterate over training set
        -- model' <- foldLoop model trainIters $ \m i -> do
        foldLoop model trainIters $ \m i -> do
            let idx  = asTensor' (trainIdx !! i :: [Int]) 
                                 ( withDevice computingDevice 
                                 . withDType Int32 
                                 $ defaultOpts )
                x    = indexSelect 0 idx trainX
                y    = indexSelect 0 idx trainY
                y'   = net m x
                loss = mseLoss y y'
                                
            when (i == trainIters) $ do
                putStrLn $ "Epoch: " ++ show epoch 
                        ++ " | Training Loss (MSE): " 
                        ++ show (asValue loss :: Double)

            -- Backprop
            (m', _) <- runStep m optim loss learningRate
            return m'
        
        -- Iterate over validation set
        -- model'' <- foldLoop model' validIters $ \m i -> do
        --     let idx  = asTensor' (trainIdx !! i :: [Int]) 
        --                          ( withDevice computingDevice 
        --                          . withDType Int32 
        --                          $ defaultOpts )
        --         x    = indexSelect 0 idx validX 
        --         y    = indexSelect 0 idx validY
        --         y'   = net m x
        --         loss = l1Loss ReduceMean y y'

        --     when (i == validIters) $ do
        --         putStrLn $ "Epoch: " ++ show epoch 
        --               ++  " | Validation Loss (MAE): " 
        --               ++ show (asValue loss :: Double)

        --     return m

        -- return model'

    -- Save final model
    save (toDependent <$> flattenParameters trainedModel) ptFile
    saveParams (flattenParameters trainedModel) ptFile

    return trainedModel
    where opts            = datasetOpts 25
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          numX            = length paramsX
          numY            = length paramsY
          numEpochs       = 42
          batchSize       = 2000
          learningRate    = asTensor' ( 1.0e-3 :: Double )
                                      ( withDType Double
                                      . withDevice computingDevice
                                      $ defaultOpts )
          ptFile          = "./models/prehsept/model.pt"

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------

-- | Operating Point Data
data OP = OP {opdev :: Device, numBatches :: Int, opData :: OPData}

instance Dataset IO OP Int (Tensor, Tensor)
    where getItem OP {..} ix = pure ( toDevice opdev . toDType Double 
                                                   . (!! ix) . inputs $ opData
                                    , toDevice opdev . toDType Double 
                                                   . (!! ix) . outputs $ opData )
          keys OP {..} = S.fromList [ 0 .. (numBatches - 1) ]

-- | Operating Point Dataset
data OPData = OPData { inputs :: [Tensor], outputs :: [Tensor] }

-- | Convenience utility functions for quickly getting a double gpu Tensor
-- toTensor :: a -> Tensor
toTensor a = asTensor' a (withDType Double . withDevice computingDevice $ defaultOpts)

type SVRet a = IO (Either NcError (SV.Vector a))

-- | Reads Data from NetCDF
getDataFromNC :: String -> [String] -> IO (M.Map String [Double])
getDataFromNC fileName params = do
    Right ncFile <- openFile fileName
    vals <- forM params 
                 (\param -> do
                    let (Just var) = ncVar ncFile param
                    Right vec <- get ncFile var :: SVRet CDouble
                    let val = map realToFrac (SV.toList vec)
                    return val)
    return (M.fromList $ zip params vals)

-- | Randomly Shuffles data
shuffleData :: M.Map String [Double] -> IO (M.Map String [Double])
shuffleData dataMap = do M.fromList . zip params . L.transpose
                    <$> (evalRandIO . shuffleM . L.transpose 
                                    . mapMaybe (`M.lookup` dataMap) 
                                    $ params)
    where params = M.keys dataMap

-- | Round to n digits
roundn :: Double -> Int -> Double
roundn d n = fromInteger (round $ d * (10^n)) / (10.0^^n)

-- | Sort data based on n-th element
sortData :: Int -> M.Map String [Double] -> M.Map String [Double]
sortData n m = M.fromList . zip p . L.transpose . L.sortBy f 
             . L.transpose . mapMaybe (`M.lookup` m) $ p
    where p = M.keys m
          f = compare `on` (!! n)

-- | Filter a datamap
filterData :: ([Double] -> Bool) -> M.Map String [Double] -> M.Map String [Double]
filterData f m = M.fromList . zip p . L.transpose . L.filter f 
               . L.transpose . mapMaybe (`M.lookup` m) $ p
    where p = M.keys m

-- | Split data into training and validation set according to ratio
splitData :: M.Map String [Double] -> Double 
          -> (M.Map String [Double], M.Map String [Double])
splitData m s = ( M.map (take numTrainSamples) m
                     , M.map (drop numTrainSamples) m )
    where (Just numSamples) = fmap length (M.lookup (head (M.keys m)) m)
          numTrainSamples   = floor . (* s) . fromIntegral $ numSamples

-- | Split data into features and targets
xyData :: M.Map String [Double] -> [String] -> [String] -> (Tensor, Tensor)
xyData m x y = ( toTensor' (mapMaybe (`M.lookup` m) x :: [[Double]])
               , toTensor' (mapMaybe (`M.lookup` m) y :: [[Double]]) )
    where toTensor' = transpose (Dim 0) (Dim 1) . toTensor

-- | Applys `ln|x|` to each feature x of the given tensor. Where `x` can be
-- | masked with an [Int].
transformData :: [Int] -> Tensor -> Tensor
transformData m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.log
                  . (+ toTensor (1 :: Double))
                  . Torch.abs 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor' (m :: [Int]) (withDType Int32 . withDevice computingDevice $ defaultOpts)
          t' = t * (1 - m')

-- | Applys `e^x` to each column x of the given tensor. Where `x` can be masked
-- | with [Int].
transformData' :: [Int] -> Tensor -> Tensor
transformData' m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.exp
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor' (m :: [Int]) (withDType Int32 . withDevice computingDevice $ defaultOpts)
          t' = t * (1 - m')

-- | Scale data such that x ∈ [a;b]. Returns a tuple, with the scaled Tensor
-- | and a Tensor [2,num-features] with min and max values per feature for
-- | unscaling.
scaleData :: Double -> Double -> Tensor -> (Tensor, Tensor)
scaleData a b x = ( ((x - xMin) * (b' - a')) / (xMax - xMin)
                  , Torch.cat (Dim 0) [ Torch.reshape [1,-1] xMin
                                      , Torch.reshape [1,-1] xMax ])
    where (xMin, _) = Torch.minDim (Dim 0) RemoveDim x
          (xMax, _) = Torch.maxDim (Dim 0) RemoveDim x
          a' = toTensor (a :: Double)
          b' = toTensor (b :: Double)
          
-- | Un-Scale data where x ∈ [a;b] for a given min and max.
scaleData' :: Double -> Double -> Tensor -> Tensor -> Tensor
scaleData' a b s y = ((y - a') / (b' - a') * (xMax - xMin)) + xMin
    where xMin = Torch.select 0 0 s
          xMax = Torch.select 0 1 s
          a' = toTensor (a :: Double)
          b' = toTensor (b :: Double)

-- | 'preprocessData' is a convenience function to clean up main.
-- | Arguments:
-- |    computing device :: Device
-- |    lower limit scaled :: FLoat
-- |    upper limit scaled :: FLoat
-- |    transformation mask X :: [Int]
-- |    transformation mask Y :: [Int]
-- |    parameters X :: [String]
-- |    parameters Y :: [String]
-- |    train test split ratio :: Double
-- |    NetCDF Data map as obtained by `getDataFromNC`
-- |    batch size :: Int
-- |    target computing device :: Device
-- | Returns a tuple  split into train and validation sets. Additionaly scalers
-- | are returned for unscaling the data later.
-- |    -> (trainingData, validationData, scalerX, scalerY)
preprocessData :: Double -> Double -> [Int] -> [Int] -> [String] -> [String] 
                -> Double -> M.Map String [Double] -> Int
                -> (OPData, OPData, Net -> Tensor -> Tensor)
preprocessData lo hi mX mY pX pY splt dat bs
        = (trainData, validData, predict)
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
          trainData                  = OPData (split bs (Dim 0) . toDevice computingDevice
                                                                . toDType Double 
                                                                $ trainX) 
                                              (split bs (Dim 0) . toDevice computingDevice 
                                                                . toDType Double 
                                                                $ trainY)
          validData                  = OPData (split bs (Dim 0) . toDType Double 
                                                                $ validX) 
                                              (split bs (Dim 0) . toDType Double 
                                                                $ validY)
          predict :: Net -> Tensor -> Tensor
          predict m x = transformData' mY . scaleData' lo hi scalerY . net m 
                      . fst . scaleData lo hi . transformData mX $ x

-- | 'preprocessData' is a convenience function to clean up main.
-- | Arguments:
-- |    lower limit scaled :: FLoat
-- |    upper limit scaled :: FLoat
-- |    transformation mask X :: [Int]
-- |    transformation mask Y :: [Int]
-- |    parameters X :: [String]
-- |    parameters Y :: [String]
-- |    train test split ratio :: Double
-- |    NetCDF Data map as obtained by `getDataFromNC`
-- | Returns a tuple of tensors, split into X and Y parametrs and split into
-- |    train and validation sets. Additionaly scalers are returned for unscaling
-- |    the data later.
-- |    -> (trainX, trainY, validX, validY, scalerX, scalerY)
preprocessData' :: Double -> Double -> [Int] -> [Int] -> [String] -> [String] 
               -> Double -> M.Map String [Double]
               -> (Tensor, Tensor, Tensor, Tensor, Net -> Tensor -> Tensor)
preprocessData' lo hi mX mY pX pY s d = ( trainX, trainY, validX, validY
                                       , predict)
    where (rawTrain, rawValid)       = splitData d s
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
          predict :: Net -> Tensor -> Tensor
          predict m x = transformData' mY . scaleData' lo hi scalerY . net m 
                      . fst . scaleData lo hi . transformData mX $ x

------------------------------------------------------------------------------
-- VALIDATION
------------------------------------------------------------------------------

-- valid :: M.Map String [Double] -> (Tensor -> Tensor) -> IO ()
-- valid d p = do
-- 
--     return ()
--     where 
--         params = M.keys d
--         Just vds = L.elemIndex "Vds" params
--         Just vgs = L.elemIndex "Vgs" params
--         Just vbs = L.elemIndex "Vbs" params
--         Just gm  = L.elemIndex "gmoverid" params
--         Just l   = L.elemIndex "L" params
--         Just ll  = (!!3) . L.nub <$> M.lookup "L" d
--         Just w   = L.elemIndex "W" params
--         Just ww  = (!!3) . L.nub <$> M.lookup "W" d
--         fd = \f -> (roundn (f !! vds) 2 == 1.65) 
--                  && ((f !! l) == ll) 
--                  && ((f !! w) == ww) 
--                  && (roundn (f !! vbs) 2 == 0.00)
--   
--         traceData   = sortData gm . filterData fd $ d
--         Just traceX = M.lookup "gmoverid" traceData
--         Just traceY = M.lookup "idoverw" traceData
--         paramsX     = ["gmoverid", "fug", "Vds", "Vbs"]
--         paramsY     = ["idoverw", "L", "W", "gdsoverw", "Vgs"]
--         xs      = transpose (Dim 0) (Dim 1) . asTensor 
--                     $ (mapMaybe (`M.lookup` traceData) paramsX :: [[Double]])
--         ys     = asValue (p xs) :: [[Double]]
--         traceY'     = (!! 0) . L.transpose $ ys
--         plotFile    = "/home/uhlmanny/Workspace/plots/valid.html"

plotData :: IO ()
plotData = do
    rawData <- getDataFromNC ncFileName (paramsX ++ paramsY)

    let params   = M.keys rawData
        Just vds = L.elemIndex "Vds" params
        Just vgs = L.elemIndex "Vgs" params
        Just vbs = L.elemIndex "Vbs" params
        Just gm  = L.elemIndex "gmoverid" params
        Just l   = L.elemIndex "L" params
        Just ll  = (!!3) . L.nub <$> M.lookup "L" rawData
        Just w   = L.elemIndex "W" params
        Just ww  = (!!3) . L.nub <$> M.lookup "W" rawData
        fd = (\d -> (roundn (d !! vds) 2 == 1.65) 
                 && ((d !! l) == ll) 
                 && ((d !! w) == ww) 
                 && (roundn (d !! vbs) 2 == 0.00))

    let traceData   = sortData gm . filterData fd $ rawData
        Just traceX = M.lookup "gmoverid" traceData
        Just traceY = M.lookup "idoverw" traceData
        traceXY :: [(Double, Double)]
        traceXY = zip (map realToFrac traceX) (map realToFrac traceY)
    
    let axis = PAxis [AxValues (Numbers (map fst traceXY))]
        enc  = encoding . position X [ PName xlabel
                                     , PmType Quantitative, axis ]
                        . position Y [ PName ylabel
                                     , PmType Quantitative
                                     , PScale [SType ScLog] ]
                        . color [ MName "Lines", MmType Nominal ]
        dt = foldl (\sum' (x, y) ->
                        sum' . dataRow [ (xlabel, Number x)
                                       , (ylabel, Number y)
                                       , ("Lines", Str "Truth") ] 
                   ) (dataFromRows []) traceXY
    
    toHtmlFile plotFile $ toVegaLite [ dt []
                                     , mark Line []
                                     , enc []
                                     , height 1000
                                     , width 1000 ]
                                    
    where ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
          modelFileName   = "./model.pt"
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "W", "gdsoverw", "Vgs"]
          maskX           = [0,1,0,0]
          maskY           = [1,0,1,0]
          numX            = length paramsX
          numY            = length paramsY
          plotFile        = "/home/uhlmanny/Workspace/plots/data.html"
          xlabel          = "gm/Id in 1/V"
          ylabel          = "Id/W in A/m"
