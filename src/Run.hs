{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}

-- | Module for running training
module Run where

import           System.ProgressBar
import           Lib
import           Net
import           HyperParameters
import           Data.Frame         as DF
import qualified Torch              as T
import qualified Torch.Extensions   as T
import qualified Torch.NN           as NN

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
    loss  = T.l1Loss T.ReduceSum trueY predY

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
runEpochs :: FilePath -> Int -> Int -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
          -> OpNet -> T.Adam -> IO (OpNet, T.Adam)
runEpochs path 0     _  _      _      _      _      net opt = do
    saveCheckPoint path net opt
    pure (net, opt)
runEpochs path epoch bs trainX validX trainY validY net opt = do
    (trainXs, trainYs) <- shuffleBatches bs trainX trainY
    (validXs, validYs) <- shuffleBatches bs trainX trainY

    tBar <- newProgressBar (trainStyle epoch) 10 (Progress 0 (length trainXs) ())
    (net', opt', lss) <- trainingEpoch tBar trainXs trainYs [] net opt

    putStrLn $ "\tTraining Loss: " ++ show (T.asValue $ T.mean lss :: Float)

    vBar <- newProgressBar (validStyle epoch) 10 (Progress 0 (length validXs) ())
    mae  <- validationEpoch vBar validXs validYs net' []
    
    putStrLn $ "\tValidataion Loss: " ++ show (T.asValue $ T.mean mae :: Float)

    saveCheckPoint path net' opt'

    runEpochs path epoch' bs trainX validX trainY validY net' opt'
  where
    epoch' = epoch - 1

-- | Initiate Training Run for given Args
run :: Args -> IO ()
run Args{..} = do
    putStrLn $ "Training " ++ show dev ++ " Model in " ++ show pdk ++ "."

    modelPath  <- createModelDir pdk' dev'
    dfRaw      <- DF.fromFile dir

    let vals   = T.cat (T.Dim 1) [         dfRaw ?? "gmoverid"
                                 ,         dfRaw ?? "fug"
                                 ,         dfRaw ?? "vds"
                                 ,         dfRaw ?? "vbs"
                                 , T.abs $ (dfRaw ?? "id") / (dfRaw ?? "W")
                                 ,         dfRaw ?? "L"
                                 ,         dfRaw ?? "id"
                                 ,         dfRaw ?? "region" ]
        dfRaw' = DF.dropNan $ DataFrame cols vals

    let sat    = T.eq (dfRaw' ?? "region") 2
        dfSat  = rowFilter sat dfRaw'

    dfShuff    <- DF.shuffleIO dfSat

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

    (net', opt') <- runEpochs modelPath num size trainX' validX' trainY' validY' net opt

    saveCheckPoint modelPath net' opt'

    -- !net'' <- T.toDevice T.cpu <$> noGrad net'
    !net'' <- loadCheckPoint modelPath (OpNetSpec numInputs numOutputs) num 
                    >>= noGrad . fst

    let tracePath = modelPath ++ "/" ++ dev' ++ ".pt"
        num'      = numInputs + 1
        predict x = y
          where
            !i'  = T.arange 1 num' 1 . T.withDType T.Int64 $ T.defaultOpts 
            !x'  = T.indexSelect 1 i' x
            !y'  = trafo' maskY . scale' minY maxY . forward net''
                 . scale minX maxX . trafo maskX $ x'
            !w   = T.reshape [-1,1] $ T.select 1 0 x / T.select 1 0 y'
            !l   = T.reshape [-1,1] $ T.select 1 1 y'
            !y   = T.abs $ T.cat (T.Dim 1) [ w, l ]

    traceModel dev pdk num' predict >>= saveInferenceModel tracePath

    putStrLn $ "Final Checkpoint saved at: " ++ modelPath
    putStrLn $ "Traced Model saved at: " ++ tracePath

    -- testModel dfRaw' ("id" : paramsX) paramsY predict
  where
    pdk'       = show pdk
    dev'       = show dev
    testSplit  = 0.75
    paramsX    = ["gmoverid", "fug", "Vds", "Vbs"]
    paramsY    = ["idoverW", "L"]
    cols       = paramsX ++ paramsY ++ ["id", "region"]
    numInputs  = length paramsX
    numOutputs = length paramsY
    maskX      = T.toDevice T.cpu $ boolMask' ["gmoverid", "id", "fug"] paramsX
    maskY      = T.toDevice T.cpu $ boolMask' ["idoverW", "L"] paramsY

testModel :: DF.DataFrame T.Tensor -> [String] -> [String]
          -> (T.Tensor -> T.Tensor) -> IO ()
testModel df paramsX paramsY mdl = do
    df' <- DF.sampleIO 10 False $ DF.rowFilter ((df ?? "region") `T.eq` 2) df
    let x  = DF.lookup paramsX df'
        y  = DF.DataFrame paramsY 
           $ T.cat (T.Dim 1) [ T.abs $ (df' ?? "id") / (df' ?? "idoverW")
                             , T.abs $ df' ?? "L" ]
        y' = DF.DataFrame paramsY . mdl $ DF.values x
    print x
    print y
    print y'
    pure ()
