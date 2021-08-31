module Main where

import Torch hiding (take) --, floor, repeat)

import System.Random
import Data.Maybe
import qualified Data.List as L
import Data.List.Split
import qualified Data.Map.Strict as M
-- import qualified Data.Map as M
import Control.Monad (when, forM)
-- import qualified Data.Functor as F

import Lib

main :: IO ()
main = do
    -- Loading Data from NetCDF
    rawData         <- getDataFromNC ncFileName (paramsX ++ paramsY)
    shuffledData    <- shuffleData rawData

    -- Split shuffle and divide data
    let (trainData, validData) = splitData shuffledData trainSplit
        (trainX, trainY) = xyData trainData paramsX paramsY
        (validX, validY) = xyData validData paramsX paramsY

    -- Normalize and Scale Data

    -- Neural Network Setup
    initModel' <- sample $ NetSpec numX numY
    let initModel = toDevice gpu initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    let numTrainSamples = head (shape trainX)
        trainIdx        = chunksOf batchSize [0 .. numTrainSamples]
        numIters        = numEpochs * length trainIdx

    trainedModel <- foldLoop initModel numIters 
                  $ \model iter -> do
                      let i     = mod (iter - 1) numEpochs
                          idx   = trainIdx !! i
                          x     = toDevice gpu . toDType Float 
                                . indexSelect' 0 idx $ trainX 
                          y     = toDevice gpu . toDType Float 
                                . indexSelect' 0 idx $ trainY
                          y'    = net model x
                          loss  = mseLoss y y'
                      
                      (model', _) <- runStep model optim loss learningRate

                    -- Validation Step after Each Epoch
                      when (i == 0) $ do
                          putStrLn $ "Epoch: " ++ show (fromIntegral numEpochs / fromIntegral iter) ++ " | Loss: " ++ show loss
                          -- putStrLn "Validation Step"

                      return model'

    putStrLn $ "Train X shape: " ++ show (shape trainX)
    -- putStrLn $ "Y shape: " ++ show (shape dataY)
    return ()

    where 
        ncFileName      = "/home/ynk/workspace/data/xh035-nmos.nc"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        trainSplit      = 0.8
        numX            = 4
        numY            = 4
        numEpochs       = 42
        batchSize       = 2000
        learningRate    = 1.0e-3
        gpu             = Device CUDA 0
        getIdx          = take batchSize . concat . Prelude.repeat

{-
main :: IO ()
main = do
    initModel' <- sample $ NetSpec numX numY
    let initModel = toDevice gpu initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    trainedModel <- foldLoop initModel numIters $ \model iter -> do
        x <- randIO' [batchSize, numX]
             F.<&> toDType Float . toDevice gpu

        let y    = genData x
            y'   = net model x
            loss = mseLoss y y'

        when (mod iter 100 == 0) $ do
            putStrLn $ "Iter: " ++ show iter ++ " | Loss: " ++ show loss

        (model', _) <- runStep model optim loss 1e-3
        return model'

    putStrLn "Done"
    return ()
  where
    gpu       = Device CUDA 0
    numX      = 4
    numY      = 4
    numIters  = 10000
    batchSize = 2000
    trainLoss = mseLoss 
    validLoss = l1Loss 
-}
