{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | Module for running training
module Run where

import           System.ProgressBar
import           Lib
import           Net
import           Data.Frame as DF
import qualified Torch      as T
import qualified Torch.NN   as NN

------------------------------------------------------------------------------
-- Utility and Helpers
------------------------------------------------------------------------------

-- | Filter Datapoints not in saturation
satMask :: Device -> DataFrame T.Tensor -> T.Tensor
satMask NMOS df = T.logicalAnd ((df ?? "Vgs") `T.gt` (df ?? "vth"))
                . T.logicalAnd ((df ?? "Vds") `T.gt` 
                        ((df ?? "Vgs") - (df ?? "vth")))
                $ (0.0 `T.lt` ((df ?? "Vgs") - (df ?? "vth")))
satMask PMOS df = T.logicalAnd (T.abs (df ?? "Vgs") `T.gt` T.abs (df ?? "vth"))
                . T.logicalAnd (T.abs (df ?? "Vds") `T.gt` 
                        (T.abs (df ?? "Vgs") - T.abs (df ?? "vth")))
                $ (0.0 `T.lt` (T.abs (df ?? "Vgs") - T.abs (df ?? "vth")))

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Run one Update Step
trainStep :: (OpNet -> T.Tensor -> T.Tensor) -> T.Tensor -> T.Tensor 
          -> OpNet -> T.Adam -> IO (OpNet, T.Adam)
trainStep predict trueX trueY net opt = T.runStep net opt loss 0.001
  where
    predY = predict net trueX
    loss  = T.mseLoss trueY predY

-- | Run through all Batches performing an update for each
trainingEpoch :: ProgressBar s -> (OpNet -> T.Tensor -> T.Tensor) -> [T.Tensor] 
              -> [T.Tensor] -> OpNet -> T.Adam -> IO (OpNet, T.Adam)
trainingEpoch _ _         _      []   net opt = pure (net, opt)
trainingEpoch _ _         []     _    net opt = pure (net, opt)
trainingEpoch bar predict (x:xs) (y:ys) net opt = do
        incProgress bar 1
        trainStep predict x' y' net opt >>= trainingEpoch' 
  where
    x' = T.toDevice compDev x
    y' = T.toDevice compDev y
    trainingEpoch' = uncurry $ trainingEpoch bar predict xs ys

------------------------------------------------------------------------------
-- Validation
------------------------------------------------------------------------------

-- | Run one Update Step
validStep :: (OpNet -> T.Tensor -> T.Tensor) -> T.Tensor -> T.Tensor 
          -> OpNet -> IO T.Tensor
validStep predict trueX trueY net = T.detach loss
  where
    predY = predict net trueX
    loss  = T.l1Loss T.ReduceMean trueY predY

-- | Run through all Batches performing an update for each
validationEpoch :: ProgressBar s -> (OpNet -> T.Tensor -> T.Tensor) 
                -> [T.Tensor] -> [T.Tensor] -> OpNet -> [T.Tensor] 
                -> IO T.Tensor
validationEpoch _ _         _      [] _ losses = pure $ T.cat (T.Dim 0) losses
validationEpoch _ _         []     _  _ losses = pure $ T.cat (T.Dim 0) losses
validationEpoch bar predict (x:xs) (y:ys) net losses = do
        incProgress bar 1
        validStep predict x' y' net >>= 
                validationEpoch bar predict xs ys net . (:losses) 
  where
    x' = T.toDevice compDev x
    y' = T.toDevice compDev y

------------------------------------------------------------------------------
-- Running
------------------------------------------------------------------------------

-- | Run Training and Validation for a given number of Epochs
runEpochs :: FilePath -> Int -> [T.Tensor] -> [T.Tensor] -> [T.Tensor] 
          -> [T.Tensor] -> (OpNet -> T.Tensor -> T.Tensor) -> OpNet -> T.Adam
          -> IO (OpNet, T.Adam)
runEpochs path 0     _       _       _       _       _       net opt = do
    saveCheckPoint path net opt
    pure (net, opt)
runEpochs path epoch trainXs validXs trainYs validYs predict net opt = do

    tBar <- newProgressBar trainStyle 10 (Progress 0 (length trainXs) ())
    (net', opt') <- trainingEpoch tBar predict trainXs trainYs net opt

    vBar <- newProgressBar validStyle 10 (Progress 0 (length validXs) ())
    _            <- validationEpoch vBar predict validXs validYs net' []

    saveCheckPoint path net' opt'

    runEpochs path epoch' trainXs validXs trainYs validYs predict net' opt'
  where
    epoch' = epoch - 1

-- | Initiate Training Run for given Args
run :: Args 
      -> IO ()
run Args{..} = do
    path <- createModelDir pdk' dev'
    df'  <- DF.fromFile dir

    let vals'   = T.cat (T.Dim 1) [ T.abs $  df' ?? "M0.m1:gmoverid"
                                  , T.abs $ (df' ?? "M0.m1:id")  / (df' ?? "W")
                                  , T.abs $ (df' ?? "M0.m1:gds") / (df' ?? "W")
                                  , T.abs $  df' ?? "M0.m1:fug"
                                  ,          df' ?? "M0.m1:vds"
                                  ,          df' ?? "M0.m1:vgs"
                                  ,          df' ?? "M0.m1:vbs"
                                  ,          df' ?? "M0.m1:vth"
                                  ,          df' ?? "M0.m1:id"
                                  ,          df' ?? "W"
                                  ,          df' ?? "L"
                                  ]

    vals <- T.detach vals' >>= T.clone

    let df''      = dropNan $ DataFrame cols vals
        minX    = fst . T.minDim (T.Dim 0) T.RemoveDim . values 
                . DF.lookup paramsX $ df''
        maxX    = fst . T.maxDim (T.Dim 0) T.RemoveDim . values 
                . DF.lookup paramsX $ df''
        minY    = fst . T.minDim (T.Dim 0) T.RemoveDim . values 
                . DF.lookup paramsY $ df''
        maxY    = fst . T.maxDim (T.Dim 0) T.RemoveDim . values 
                . DF.lookup paramsY $ df''

    let sat   = satMask dev df''
        sat'  = T.logicalNot sat
        nSat' = (`div` 4) . head . T.shape . T.nonzero $ sat
        dfSat = rowFilter sat df''

    dfSat' <- DF.sampleIO nSat' False $ rowFilter sat' df''
    df     <- DF.shuffleIO $ DF.concat [dfSat, dfSat']

    net <- T.sample $ OpNetSpec (length paramsX) (length paramsX)
    let opt = T.mkAdam 0 0.9 0.999 $ NN.flattenParameters net
    
    let predict = predictor minX maxX minY maxY maskX maskY

    let (trainX, validX, trainY, validY) = trainTestSplit paramsX paramsY 
                                                          testSplit df
        batchesX  = T.split size (T.Dim 0) trainX
        batchesY  = T.split size (T.Dim 0) trainY
        batchesX' = T.split size (T.Dim 0) validX
        batchesY' = T.split size (T.Dim 0) validY
        traceData = T.cat (T.Dim 0) [trainX, validX]
        tracePath = path ++ "trace.pt"

    (net', opt') <- runEpochs path num batchesX batchesX' batchesY batchesY' 
                              predict net opt

    saveCheckPoint path net' opt'
    traceModel dev pdk traceData (predict net') >>= saveModel tracePath

    pure ()
  where
    pdk'      = show pdk
    dev'      = show dev
    testSplit = 0.8
    cols      = [ "gmoverid", "idoverw", "gdsoverw", "fug"
                , "Vds", "Vgs", "Vbs", "vth", "id" ]
    paramsX   = ["gmoverid", "fug", "Vds", "Vbs"]
    paramsY   = ["idoverw", "L", "gdsoverw", "Vgs"]
    maskX     = boolMask' ["fug"] paramsX
    maskY     = boolMask' ["idoverw", "gdsoverw"] paramsY
