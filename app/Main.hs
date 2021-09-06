module Main where

import Torch hiding (take, floor) 

import Data.Maybe
import Data.List.Split
import Control.Monad (when, forM)
import Control.Monad.Cont (runContT)

import Pipes
import qualified Pipes.Prelude as P

import Lib

trainLoop :: Optimizer o => Net -> o -> Float -> Int -> Int 
          -> ListT IO (Tensor, Tensor) -> IO Net
trainLoop model optim lr logRate epoch = P.foldM step begin done 
                                                 . enumerateData
    where step :: Net -> ((Tensor, Tensor), Int) -> IO Net
          step m ((x, y), i) = do
                let y' = net m x
                    loss = mseLoss y y'

                when (i `mod` logRate == 0) $ do
                    putStrLn $ show epoch ++ " / " ++ show i 
                            ++ " | Loss: " ++ show (asValue loss :: Float)

                (m', _) <- runStep m optim loss (asTensor (lr :: Float))
                pure m'

          done = pure
          begin = pure model

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
    let numBatches = length (inputs trainData)
        trainSet = OP {dev = dev, numBatches = numBatches, opData = trainData}
        validSet = OP {dev = dev, numBatches = numBatches, opData = validData}
    
    -- Neural Network Setup
    initModel' <- sample $ NetSpec numX numY

    let initModel = toDevice dev initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    -- Training
    mdl <- foldLoop initModel numEpochs (\model epoch ->
                runContT (streamFromMap (datasetOpts 100) trainSet) $ 
                    trainLoop model optim learningRate numBatches epoch . fst)

    -- model = initModel
    -- epoch = 0

    -- trainedModel <- foldLoop initModel numEpochs 
    --               $ \model epoch -> do

    --                     let getBatch :: Int -> [Int] -> Tensor -> Tensor
    --                         getBatch = (\d r t -> toDevice dev . toDType Float 
    --                                             $ indexSelect' d r t)
    --                         fwd :: [Int] -> Tensor
    --                         fwd = (\idx -> let x = getBatch 0 idx trainX 
    --                                            y = getBatch 0 idx trainY
    --                                            z = net model x
    --                                        in mseLoss y z)

    --                     let avgLoss = toDType Float . mean . cat (Dim 0) 
    --                                 . map (reshape [-1,1] . fwd) $ batchIdx

    --                     (model', _) <- runStep model optim avgLoss learningRate

    --                     let vx = toDevice dev . toDType Float $ validX 
    --                         vy = toDevice dev . toDType Float $ validY
    --                         vz = net model vx
    --                         vl = toDevice dev . toDType Float 
    --                            $ l1Loss ReduceMean vy vz
    --                     
    --                     putStrLn $ "Epoch: " ++ show epoch  
    --                             ++ " | Training Loss (MSE): " ++ show (asValue avgLoss :: Float) 
    --                             -- ++ " Validation Loss (MAE): " ++ show (asValue vl :: Float)

    --                     return model'

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
        dev             = Device CUDA 1
        -- dev             = Device CPU 0
        learningRate = 5.0e-4
