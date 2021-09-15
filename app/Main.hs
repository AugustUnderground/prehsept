module Main where

import qualified Data.Map as M

import Lib

main :: IO ()
main = do
    -- Loading Data from NetCDF
    rawData     <- getDataFromNC ncFileName (paramsX ++ paramsY)
    sampledData <- M.map (take numSamples) <$> shuffleData rawData

    -- Train with dataset/stream
    -- let (trainData, validData, predict) 
    --         = preprocessData lower upper maskX maskY 
    --                          paramsX paramsY trainSplit 
    --                          sampledData batchSize
    -- model <- trainNet trainData validData

    -- Train with raw Tensors
    let (trainX, trainY, validX, validY, predict)
            = preprocessData' lower upper maskX maskY 
                              paramsX paramsY trainSplit 
                              sampledData
    model <- trainNet' (trainX, trainY) (validX, validY)

    putStrLn "Done!"
    return ()

    where ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          maskX           = [0,1,0,0]
          maskY           = [1,0,1,0]
          trainSplit      = 0.8
          lower           = 0
          upper           = 1
          numSamples      = 666666
          batchSize       = 2000
