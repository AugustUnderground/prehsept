module Main where

import qualified Data.Map as M
import Data.Time.Clock

import Lib

main :: IO ()
-- main = train
main = eval "../models/prehsept/xh035-nmos-2021-09-2016-37-46-687795907.pt"

train :: IO ()
train = do
    --- Get current timestamp for file name
    ts <- concatMap (\p -> [if c == ':' || c == '.' then '-' else c | c <- p]) 
        . init . words . show <$> getCurrentTime

    --- Loading Data from NetCDF
    rawData     <- getDataFromNC ncFileName (paramsX ++ paramsY ++ ["W"])
    sampledData <- M.map (take numSamples) <$> shuffleData rawData

    ------ TRAINING
    --- Process training data
    let (trainData, validData, predict) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit 
                             sampledData batchSize
    
    --- Train with dataset/stream
    model <- trainNet trainData validData

    --- Save trained model
    saveNet model (modelPrefix ++ ts ++ ".pt")

    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"

    where technology      = "xh035"
          deviceType      = "nmos"
          ncFileName      = "/home/uhlmanny/Workspace/data/" 
                          ++ technology ++ "-" ++ deviceType  ++ ".nc"
          modelPrefix     = "../models/prehsept/" ++ technology 
                          ++ "-" ++ deviceType ++ "-"
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          numX            = length paramsX
          numY            = length paramsY
          maskX           = [0,1,0,0]
          maskY           = [1,0,1,0]
          trainSplit      = 0.8
          lower           = 0
          upper           = 1
          numSamples      = 500000
          --numSamples      = 200000
          batchSize       = 2000

eval :: String -> IO ()
eval ptFile = do
    --- Loading Data from NetCDF
    rawData     <- getDataFromNC ncFileName (paramsX ++ paramsY ++ ["W"])

    let (trainData, validData, predict) 
            = preprocessData lower upper maskX maskY 
                             paramsX paramsY trainSplit 
                             rawData batchSize

    --- Load trained model
    model' <- loadNet ptFile numX numY

    --- Extract the prediction method
    let predict' = predict model'

    --- Make predictions and plot vs. ground truth
    plotPredictionVsTruth rawData predict' "gmoverid" "idoverw"

    ------ DONE
    putStrLn "``'-.,_,.-'``'-.,_,.='``'-., DONE ,.-'``'-.,_,.='``'-.,_,.='``"
 
    where technology      = "xh035"
          deviceType      = "nmos"
          ncFileName      = "/home/uhlmanny/Workspace/data/" 
                          ++ technology ++ "-" ++ deviceType  ++ ".nc"
          paramsX         = ["gmoverid", "fug", "Vds", "Vbs"]
          paramsY         = ["idoverw", "L", "gdsoverw", "Vgs"]
          numX            = length paramsX
          numY            = length paramsY
          maskX           = [0,1,0,0]
          maskY           = [1,0,1,0]
          trainSplit      = 0.8
          lower           = 0
          upper           = 1
          batchSize       = 2000
