module Main where

import Torch hiding (take, floor) 

import Control.Monad (when, forM)
import Control.Monad.Cont (runContT)

import Graphics.Gnuplot.Simple
import qualified Graphics.Gnuplot.Terminal.SVG as SVG

import Lib

main :: IO ()
main = do
    -- Loading Data from NetCDF
    rawData         <- getDataFromNC ncFileName (paramsX ++ paramsY)
    shuffledData    <- shuffleData rawData

    -- Process data
    let (trainData, validData, scalerX, scalerY) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit shuffledData
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

    plotList [ Key Nothing
             , terminal (SVG.cons "./plot.svg")
             ] $ zip [1 .. length loss] loss

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
        numWorkers      = 25
        dev             = Device CUDA 1
        -- dev             = Device CPU 0
        learningRate    = 1.0e-3
