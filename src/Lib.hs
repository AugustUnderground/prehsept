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
           , train
           , valid
           , preprocessData
           , OP (..)
           , OPData (..)
           , getDataFromNC
           , shuffleData
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
trainLoop :: Optimizer o => Net -> o -> Tensor -> ListT IO (Tensor, Tensor) 
          -> IO (Net, Float)
trainLoop model optim lr = P.foldM step begin done . enumerateData
    where step :: (Net, Float) -> ((Tensor, Tensor), Int) -> IO (Net, Float)
          step (m, l) ((x, y), i) = do
                putStrLn $ "X device: " ++ show (device x)
                putStrLn $ "Y device: " ++ show (device y)
                let y' = net m x
                    l' = mseLoss y y'
                    l'' = l + (asValue l' :: Float)

                (m', _) <- runStep m optim l' lr -- (asTensor (lr :: Float))
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
train :: OPData -> OPData -> IO Net
train trainData validData = do
    -- Neural Network Setup
    initModel' <- sample $ NetSpec numX numY

    let initModel = toDevice dev initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    foldLoop initModel numEpochs $ \model epoch -> do
        foldLoop model numTrainBatches $ \m b -> do
            let (x,y) = trainSet !! b
                y'    = net m x
                l     = mseLoss y y'
            (m', _) <- runStep m optim l learningRate

            putStrLn $ show epoch ++  " | Training Loss (MSE): " 
                    ++ show l
            return m'

    -- foldLoop initModel numEpochs $ \m e -> do
    --     (m', l') <- runContT (streamFromMap opts trainSet)
    --                     $ trainLoop m optim learningRate . fst
    --     putStrLn $ show e ++  " | Training Loss (MSE): " 
    --             ++ show (l' / fromIntegral numTrainBatches)
    --     return m'

    --     -- (_, vl)  <- runContT (streamFromMap opts validSet)
    --     --                 $ validLoop m' . fst
    --     -- putStrLn $ "    +-> Validation Loss (MAE): " 
    --     --         ++ show (vl / fromIntegral numValidBatches)

    where 

        opts            = datasetOpts 25
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        numX            = length paramsX
        numY            = length paramsY
        numEpochs       = 42
        dev             = Device CUDA 1
        -- dev             = Device CPU 0
        learningRate    = toDevice dev $ asTensor (1.0e-3 :: Float)
        numTrainBatches = length (inputs trainData)
        trainSet        = zip (inputs trainData) (outputs trainData)
        -- trainSet        = OP { opdev = dev, numBatches = numTrainBatches
        --                      , opData = trainData }
        -- numValidBatches = length (inputs validData)
        -- validSet        = OP { opdev = dev, numBatches = numValidBatches
        --                      , opData = validData }

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------

data OP = OP {opdev :: Device, numBatches :: Int, opData :: OPData}

instance Dataset IO OP Int (Tensor, Tensor)
    where getItem OP {..} ix = pure ( toDevice opdev . toDType Float 
                                                   . (!! ix) . inputs $ opData
                                    , toDevice opdev . toDType Float 
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
                -> (OPData, OPData, Net -> Tensor -> Tensor)
preprocessData lo hi mX mY pX pY splt dat bs dev 
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
          trainData                  = OPData (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ trainX) 
                                              (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ trainY)
          validData                  = OPData (split bs (Dim 0) . toDType Float 
                                                                $ validX) 
                                              (split bs (Dim 0) . toDType Float 
                                                                $ validY)
          predict :: Net -> Tensor -> Tensor
          predict m x = transformData' mY . scaleData' lo hi scalerY . net m 
                      . fst . scaleData lo hi . transformData mX $ x

------------------------------------------------------------------------------
-- VALIDATION
------------------------------------------------------------------------------

valid :: M.Map String [Float] -> (Tensor -> Tensor) -> IO ()
valid d p = do

    T.writeFile plotFile . renderText . doctypehtml_ $ do
                        head_ $ do meta_ [charset_ "utf-8"] 
                                   plotlyCDN
                        body_ . toHtml $ plotly "myDiv" [trace, trace'] 
                                        & layout %~ (xaxis ?~ (defAxis 
                                            & axistitle ?~ "gm/Id [1/V]"))
                                        & layout %~ (yaxis ?~ (defAxis 
                                            & axistype  ?~ Log
                                            & axistitle ?~ "Id/W [A/m]"))
                                        & layout %~ title ?~ "Drain Current Densitiy vs. Efficiency"

    return ()
    where 
        params = M.keys d
        Just vds = L.elemIndex "Vds" params
        Just vgs = L.elemIndex "Vgs" params
        Just vbs = L.elemIndex "Vbs" params
        Just gm  = L.elemIndex "gmoverid" params
        Just l   = L.elemIndex "L" params
        Just ll  = (!!3) . L.nub <$> M.lookup "L" d
        Just w   = L.elemIndex "W" params
        Just ww  = (!!3) . L.nub <$> M.lookup "W" d
        fd = \f -> (roundn (f !! vds) 2 == 1.65) 
                 && ((f !! l) == ll) 
                 && ((f !! w) == ww) 
                 && (roundn (f !! vbs) 2 == 0.00)
  
        traceData   = sortData gm . filterData fd $ d
        Just traceX = M.lookup "gmoverid" traceData
        Just traceY = M.lookup "idoverw" traceData
        paramsX     = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY     = ["idoverw", "L", "W", "gdsoverw", "Vgs"]
        xs      = transpose (Dim 0) (Dim 1) . asTensor 
                    $ (mapMaybe (`M.lookup` traceData) paramsX :: [[Float]])
        ys     = asValue (p xs) :: [[Float]]
        traceY'     = (!! 0) . L.transpose $ ys
        trace       = line (aes & x .~ fst & y .~ snd) $ zip traceX traceY
        trace'      = line (aes & x .~ fst & y .~ snd) $ zip traceX traceY'
        plotFile    = "/home/uhlmanny/Workspace/plots/valid.html"

plotData :: IO ()
plotData = do
    rawData <- getDataFromNC ncFileName (paramsX ++ paramsY)

    let params = M.keys rawData
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
        trace       = line (aes & x .~ fst & y .~ snd) $ zip traceX traceY
    

    T.writeFile plotFile . renderText . doctypehtml_ $ do
                        head_ $ do meta_ [charset_ "utf-8"] 
                                   plotlyCDN
                        body_ . toHtml $ plotly "myDiv" [trace] 
                                        & layout %~ (xaxis ?~ (defAxis 
                                            & axistitle ?~ "gm/Id [1/V]"))
                                        & layout %~ (yaxis ?~ (defAxis 
                                            & axistype  ?~ Log
                                            & axistitle ?~ "Id/W [A/m]"))
                                        & layout %~ title ?~ "Drain Current Densitiy vs. Efficiency"

                                    
    return ()
    where
        ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
        modelFileName   = "./model.pt"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "W", "gdsoverw", "Vgs"]
        maskX           = [0,1,0,0]
        maskY           = [1,0,1,0]
        numX            = length paramsX
        numY            = length paramsY
        plotFile        = "/home/uhlmanny/Workspace/plots/data.html"
