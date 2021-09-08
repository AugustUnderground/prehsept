module Main where

import Data.Time
import Data.List.Split
import System.Environment
import qualified Data.Map as M
import Torch hiding (take, floor)
import Lib

main :: IO ()
main = do

    -- Current time stamp for filenames etc.
    t <- (init . (!!1) . splitOn "." . show) . utctDayTime <$> getCurrentTime
    d <- show . utctDay <$> getCurrentTime

    let timeStamp = d ++ "-" ++ t
        modelFileName = modelDir ++ "/" ++ technology ++ "-" ++ modelType 
                                 ++ "-" ++ timeStamp ++ ".pt"

    -- Loading Data from NetCDF and shuffling
    rawData         <- getDataFromNC ncFileName (paramsX ++ paramsY)
    shuffledData    <- shuffleData rawData
    let sampledData = M.map (take numSamples) shuffledData

    -- Process data (no shuffling bc no more IO)
    let (trainData, validData, eval) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit sampledData
                             batchSize dev

    model <- train trainData validData
    
    saveParams model modelFileName

    let predict :: Tensor -> Tensor
        predict = eval model

    valid rawData predict

    return ()
    where

        modelType       = "nmos"
        technology      = "xh035"
        ncFileName      = "/home/uhlmanny/Workspace/data/" ++ technology 
                            ++ "-" ++ modelType ++ ".nc"
        modelDir        = "/home/uhlmanny/Workspace/models/hs"
        paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
        paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
        maskX           = [0,1,0,0]
        maskY           = [1,0,1,0]
        numSamples      = 500000 -- 666666
        trainSplit      = 0.9
        lower           = 0
        upper           = 1
        batchSize       = 10000
        numWorkers      = 25
        dev             = Device CUDA 1
        -- dev             = Device CPU 0
