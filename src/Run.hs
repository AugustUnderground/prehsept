{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}

-- | Module for running training
module Run where

import           System.ProgressBar
import           Lib
import           Net
import           HyperParameters
import           Data.Frame       as DF
import qualified Torch            as T
import qualified Torch.Extensions as T
import qualified Torch.NN         as NN

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

    tBar <- newProgressBar (trainStyle epoch) 10 (Progress 0 (length trainXs) ())
    (net', opt', mse) <- trainingEpoch tBar trainXs trainYs [] net opt

    putStrLn $ "\tTraining Loss: " ++ show (T.mean mse)

    vBar <- newProgressBar (validStyle epoch) 10 (Progress 0 (length validXs) ())
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

    modelPath  <- createModelDir pdk' dev'
    dfRaw      <- DF.fromFile pdk dir

    let vals   = T.cat (T.Dim 1) [ T.abs $  dfRaw ?? "M0.m1:gmoverid"
                                 , T.abs $  dfRaw ?? "M0.m1:fug"
                                 ,          dfRaw ?? "M0.m1:vds"
                                 ,          dfRaw ?? "M0.m1:vbs"
                                 , T.abs $ (dfRaw ?? "M0.m1:id")  / (dfRaw ?? "W")
                                 ,          dfRaw ?? "L"
                                 , T.abs $ (dfRaw ?? "M0.m1:gds") / (dfRaw ?? "W")
                                 ,          dfRaw ?? "M0.m1:vgs"
                                 ,          dfRaw ?? "M0.m1:vth"
                                 ,          dfRaw ?? "M0.m1:id"
                                 ,          dfRaw ?? "W"
                                 ]
        dfRaw' = DF.dropNan $ DataFrame cols vals

    let sat    = satMask dev dfRaw'
        sat'   = T.logicalNot sat
        nSat'  = (`div` 4) . head . T.shape . T.nonzero $ sat
        dfSat  = rowFilter sat dfRaw'

    dfSat'     <- DF.sampleIO nSat' False $ rowFilter sat' dfRaw'

    dfShuff    <- DF.shuffleIO (DF.concat [dfSat, dfSat'])

    let dfT    = DF.dropNan 
               $ DF.union (trafo maskX <$> DF.lookup paramsX dfShuff)
                          (trafo maskY <$> DF.lookup paramsY dfShuff)
        dfX'   = DF.lookup paramsX dfT
        dfY'   = DF.lookup paramsY dfT

    let minX   = fst . T.minDim (T.Dim 0) T.RemoveDim . values $ dfX'
        maxX   = fst . T.maxDim (T.Dim 0) T.RemoveDim . values $ dfX'
        minY   = fst . T.minDim (T.Dim 0) T.RemoveDim . values $ dfY'
        maxY   = fst . T.maxDim (T.Dim 0) T.RemoveDim . values $ dfY'

    let !dfX   = scale minX maxX <$> dfX'
        !dfY   = scale minY maxY <$> dfY'
        !df    = DF.dropNan $ DF.union dfX dfY

    net        <- T.toDevice T.gpu <$> T.sample (OpNetSpec numInputs numOutputs)
    let opt    =  T.mkAdam 0 β1 β2 $ NN.flattenParameters net

    let (!trainX', !validX', !trainY', !validY') = 
                trainTestSplit paramsX paramsY testSplit df

    let trainX = T.split size (T.Dim 0) . T.toDevice T.gpu $ trainX'
        trainY = T.split size (T.Dim 0) . T.toDevice T.gpu $ trainY'
        validX = T.split size (T.Dim 0) . T.toDevice T.gpu $ validX'
        validY = T.split size (T.Dim 0) . T.toDevice T.gpu $ validY'

    (net', opt') <- runEpochs modelPath num trainX validX trainY validY net opt

    saveCheckPoint modelPath net' opt'

    !net'' <- T.toDevice T.cpu <$> noGrad net'

    let tracePath = modelPath ++ "/trace.pt"
        predict   = trafo' maskY
                  . scale' minY maxY
                  . forward net''
                  . scale minX maxX
                  . trafo maskX

    putStrLn $ "Final Checkpoint saved at: " ++ modelPath
    putStrLn $ "Traced Model saved at: " ++ tracePath

    traceModel dev pdk numInputs predict >>= saveInferenceModel tracePath
  where
    pdk'       = show pdk
    dev'       = show dev
    testSplit  = 0.8
    paramsX    = ["gmoverid", "fug", "Vds", "Vbs"]
    paramsY    = ["idoverw", "L", "gdsoverw", "Vgs"]
    cols       = paramsX ++ paramsY ++ ["vth", "id", "W"]
    numInputs  = length paramsX
    numOutputs = length paramsY
    maskX      = boolMask' ["fug"] paramsX
    maskY      = boolMask' ["idoverw", "gdsoverw"] paramsY
