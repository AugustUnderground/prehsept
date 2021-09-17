{-# LANGUAGE BangPatterns #-}

module Main where

import Debug.Trace
import qualified Data.Map as M

import Lib

main :: IO ()
main = do
    --- Loading Data from NetCDF
    rawData     <- getDataFromNC ncFileName (paramsX ++ paramsY ++ ["W"])
    sampledData <- M.map (take numSamples) <$> shuffleData rawData

    ------ TRAINING
    --- Train with dataset/stream
    let (trainData, validData, predict) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit 
                             sampledData batchSize

    -- model <- trainNet trainData validData

    --- Train with raw Tensors
    -- let (trainX, trainY, validX, validY, predict)
    --         = preprocessData' lower upper maskX maskY 
    --                           paramsX paramsY trainSplit 
    --                           sampledData
    --
    -- model <- trainNet' (trainX, trainY) (validX, validY)

    --- Save trained model
    -- saveNet model ptFile

    ------ EVALUATION
    --- Load trained model
    model' <- loadNet ptFile numX numY

    --- Extract the prediction method
    let predict' = predict model'

    --- Make predictions and plot vs. ground truth
    plotPredictionVsTruth rawData predict' "gmoverid" "idoverw"

    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"
    return ()

    where 

          ncFileName      = "/home/uhlmanny/Workspace/data/xh035-nmos.nc"
          ptFile          = "../models/prehsept/model.pt"
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          numX            = length paramsX
          numY            = length paramsY
          maskX           = [0,1,0,0]
          maskY           = [1,0,1,0]
          trainSplit      = 0.8
          lower           = 0
          upper           = 1
          numSamples      = 500000 -- 666666
          batchSize       = 2000
