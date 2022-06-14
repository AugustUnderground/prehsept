{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BangPatterns #-}

-- | Module for running training
module Run where

import           System.ProgressBar
import           Lib
import           Net
import           HyperParameters
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

-- | Process Data
process :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
process mi ma tf = scale mi ma . trafo tf

------------------------------------------------------------------------------
-- Training
------------------------------------------------------------------------------

-- | Run one Update Step
trainStep :: T.Tensor -> T.Tensor 
          -> OpNet -> T.Adam -> IO (OpNet, T.Adam, T.Tensor)
trainStep trueX trueY net opt = do
    (net', opt') <- T.runStep net opt loss α
    pure (net', opt', loss)
  where
    predY = forward net trueX
    loss  = T.mseLoss trueY predY

-- | Run through all Batches performing an update for each
trainingEpoch :: ProgressBar s -> [T.Tensor]  -> [T.Tensor] -> [T.Tensor] 
              -> OpNet -> T.Adam -> IO (OpNet, T.Adam, T.Tensor)
trainingEpoch _     _      []   losses net opt = pure (net, opt, losses')
  where
    losses' = T.cat (T.Dim 0) . map (T.reshape [-1]) $ losses
trainingEpoch _     []     _    losses net opt = pure (net, opt, losses')
  where
    losses' = T.cat (T.Dim 0) . map (T.reshape [-1]) $ losses
trainingEpoch bar (x:xs) (y:ys) losses net opt = do
        incProgress bar 1
        trainStep x y net opt >>= trainingEpoch' 
  where
    trainingEpoch' = \ (n, o, l) -> trainingEpoch bar xs ys (l:losses) n o

------------------------------------------------------------------------------
-- Validation
------------------------------------------------------------------------------

-- | Run one Update Step
validStep :: T.Tensor -> T.Tensor -> OpNet -> IO T.Tensor
validStep trueX trueY net = T.detach loss
  where
    predY = forward net trueX
    loss  = T.l1Loss T.ReduceMean trueY predY

-- | Run through all Batches performing an update for each
validationEpoch :: ProgressBar s -> [T.Tensor] -> [T.Tensor] -> OpNet 
                -> [T.Tensor] -> IO T.Tensor
validationEpoch _     _      []   _   losses = pure . T.cat (T.Dim 0) 
                                             . map (T.reshape [-1]) $ losses
validationEpoch _     []     _    _   losses = pure . T.cat (T.Dim 0) 
                                             . map (T.reshape [-1]) $ losses
validationEpoch bar (x:xs) (y:ys) net losses = do
        incProgress bar 1
        validStep x y net >>= 
                validationEpoch bar xs ys net . (:losses) 

------------------------------------------------------------------------------
-- Running
------------------------------------------------------------------------------

-- | Run Training and Validation for a given number of Epochs
runEpochs :: FilePath -> Int -> [T.Tensor] -> [T.Tensor] -> [T.Tensor] 
          -> [T.Tensor] -> OpNet -> T.Adam -> IO (OpNet, T.Adam)
runEpochs path 0     _       _       _       _       net opt = do
    saveCheckPoint path net opt
    pure (net, opt)
runEpochs path epoch trainXs validXs trainYs validYs net opt = do

    tBar <- newProgressBar trainStyle 10 (Progress 0 (length trainXs) ())
    (net', opt', mse) <- trainingEpoch tBar trainXs trainYs [] net opt

    putStrLn $ "\tTraining Loss: " ++ show (T.mean mse)

    vBar <- newProgressBar validStyle 10 (Progress 0 (length validXs) ())
    mae  <- validationEpoch vBar validXs validYs net' []
    
    putStrLn $ "\tValidataion Loss: " ++ show (T.mean mae)

    saveCheckPoint path net' opt'

    runEpochs path epoch' trainXs validXs trainYs validYs net' opt'
  where
    epoch' = epoch - 1

-- | Initiate Training Run for given Args
run :: Args 
      -> IO ()
run Args{..} = do
    putStrLn $ "Training " ++ show dev ++ " Model in " ++ show pdk ++ "."

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

    let df''    = DataFrame cols vals
        minX    = fst . T.minDim (T.Dim 0) T.RemoveDim 
                . values . DF.lookup paramsX $ df''
        maxX    = fst . T.maxDim (T.Dim 0) T.RemoveDim 
                . values . DF.lookup paramsX $ df''
        minY    = fst . T.minDim (T.Dim 0) T.RemoveDim 
                . values . DF.lookup paramsY $ df''
        maxY    = fst . T.maxDim (T.Dim 0) T.RemoveDim 
                . values . DF.lookup paramsY $ df''

    let sat   = satMask dev df''
        sat'  = T.logicalNot sat
        nSat' = (`div` 4) . head . T.shape . T.nonzero $ sat
        dfSat = rowFilter sat df''

    dfSat' <- DF.sampleIO nSat' False $ rowFilter sat' df''
    dfS    <- DF.shuffleIO $ DF.concat [dfSat, dfSat']

    let !df = DF.dropNan 
            $ DF.union (process minX maxX maskX <$> DF.lookup paramsX dfS) 
                       (process minY maxY maskY <$> DF.lookup paramsY dfS)

    net <- T.toDevice gpu <$> T.sample (OpNetSpec numX numY)

    let opt = T.mkAdam 0 β1 β2 $ NN.flattenParameters net
    
    let (!trainX, !validX, !trainY, !validY) = 
                trainTestSplit paramsX paramsY testSplit df

    let !batchesX  = T.split size (T.Dim 0) . T.toDevice gpu $ trainX
        !batchesY  = T.split size (T.Dim 0) . T.toDevice gpu $ trainY
        !batchesX' = T.split size (T.Dim 0) . T.toDevice gpu $ validX
        !batchesY' = T.split size (T.Dim 0) . T.toDevice gpu $ validY

    (net', opt') <- runEpochs path num batchesX batchesX' 
                                       batchesY batchesY' 
                              net opt

    saveCheckPoint path net' opt'

    net'' <- T.toDevice cpu <$> noGrad net'
    let predict   = predictor minX maxX minY maxY maskX maskY net''
        traceData = values $ DF.lookup paramsX dfS
        tracePath = path ++ "/trace.pt"

    traceModel dev pdk traceData predict >>= saveInferenceModel tracePath

    pure ()
  where
    pdk'      = show pdk
    dev'      = show dev
    testSplit = 0.8
    cols      = [ "gmoverid", "idoverw", "gdsoverw", "fug"
                , "Vds", "Vgs", "Vbs", "vth", "id", "W", "L" ]
    paramsX   = ["gmoverid", "fug", "Vds", "Vbs"]
    paramsY   = ["idoverw", "L", "gdsoverw", "Vgs"]
    numX      = length paramsX
    numY      = length paramsY
    maskX     = boolMask' ["fug"] paramsX
    maskY     = boolMask' ["idoverw", "gdsoverw"] paramsY
