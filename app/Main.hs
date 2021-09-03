module Main where

import Torch hiding (take, floor) 

import System.Random
import Data.Maybe
import qualified Data.List as L
import Data.List.Split
-- import qualified Data.Map.Strict as M
import Control.Monad (when, forM)

import Lib

main :: IO ()
main = do
    -- Loading Data from NetCDF
    rawData         <- getDataFromNC ncFileName (paramsX ++ paramsY)
    shuffledData    <- shuffleData rawData

    let (trainX, trainY, validX, validY, scalerX, scalerY) 
            = preprocessData lower upper maskX maskY paramsX paramsY trainSplit shuffledData

    -- Neural Network Setup
    initModel' <- sample $ NetSpec numX numY
    let initModel = toDevice dev initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    let numTrainSamples = head (shape trainX)
        trainIdx        = chunksOf batchSize [0 .. numTrainSamples]
        numIters        = numEpochs * length trainIdx

    trainedModel <- foldLoop initModel numIters 
                  $ \model iter -> do
                      let i     = mod (iter - 1) numEpochs
                          idx   = trainIdx !! i
                          x     = toDevice dev . toDType Float 
                                . indexSelect' 0 idx $ trainX 
                          y     = toDevice dev . toDType Float 
                                . indexSelect' 0 idx $ trainY
                          y'    = net model x
                          loss  = mseLoss y y'
                      
                      (model', _) <- runStep model optim loss learningRate

                    -- Validation Step after Each Epoch
                      when (i == 0) $ do
                          let vx     = toDevice dev . toDType Float $ validX 
                              vy     = toDevice dev . toDType Float $ validY
                              vy'    = net model vx
                              vLoss  = l1Loss ReduceMean vy vy'
                              epoch  = floor $ fromIntegral (iter - 1) 
                                             / fromIntegral numEpochs
                          putStrLn $ "Epoch: " ++ show epoch  
                                  ++ " | Training Loss (MSE): " ++ show (asValue loss :: Float) 
                                  ++ " Validation Loss (MAE): " ++ show (asValue vLoss :: Float)

                      return model'

    putStrLn "Done!"
    return ()

    where 
        ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        maskX           = [0,1,0,0]
        maskY           = [1,0,1,0]
        numX            = length paramsX
        numY            = length paramsY
        trainSplit      = 0.8
        lower           = 0
        upper           = 1
        numEpochs       = 42
        batchSize       = 2000
        learningRate    = 1.0e-3
        -- dev             = Device CUDA 0
        dev             = Device CPU 0
        getIdx          = take batchSize . concat . Prelude.repeat
