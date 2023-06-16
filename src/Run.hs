{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

-- | Module for running training
module Run where

import           System.ProgressBar
import           Lib
import           Net
import           HyperParameters
import           Data.Frame                as DF
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T         (negative)
import qualified Torch.NN           as NN

import           Prelude                         hiding (exp)

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
run args@Args{..} | exp       = runExp args
                  | otherwise = runBase args

-- | Experimental Training Run
runExp :: Args -> IO ()
runExp Args{..} = do
    putStrLn $ "Training " ++ show dev ++ " Model in " ++ show pdk ++ "."

    df' <- DF.fromFile dir

    let cls'   = ["id", "gm", "gds", "gmbs", "cgs", "cgd", "cgb", "csd", "cbs", "cbd"]
        cls''  = ["idoverW", "gmoverW", "gdsoverW", "gmbsoverW", "cgsoverW", "cgdoverW", "cgboverW", "csdoverW", "cbsoverW", "cbdoverW"]
        dfNorm = DF.DataFrame cls'' $ T.div (DF.values $ DF.lookup cls' df') (df' ?? "W")
        dfRaw  = DF.union df' dfNorm

    modelPath <- createModelDir pdk' dev'
    let tracePath = modelPath ++ "/" ++ dev' ++ "-exp.pt"

    let dfRaw' = DF.dropNan $ DF.lookup cols dfRaw
        dfReg  = DF.rowFilter (region (-1) dfRaw') dfRaw'

    dfShuff    <- DF.shuffleIO dfReg

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

    let predict = neg maskNeg . trafo' maskY . scale' minY maxY
                . forward net''
                . scale minX maxX . trafo maskX

    traceModel dev pdk numInputs paramsX paramsY predict >>= saveInferenceModel tracePath

    mdl'' <- unTraceModel <$> T.loadScript T.WithoutRequiredGrad tracePath

    putStrLn $ "Final Checkpoint saved at: " ++ modelPath
    putStrLn $ "Traced Model saved at: " ++ tracePath
    
    testModelExp dfRaw' paramsX paramsY mdl''
  where
    pdk'       = show pdk
    dev'       = show dev
    testSplit  = 0.75
    paramsX    = ["L", "vds", "vgs", "vbs"]
    paramsY    = [ "fug", "gmoverid"
                 , "idoverW", "gmoverW", "gdsoverW", "gmbsoverW"
                 , "vdsat", "vth"
                 , "cgsoverW", "cgdoverW", "cgboverW", "csdoverW", "cbsoverW", "cbdoverW" ]
    cols       = paramsX ++ paramsY ++ ["region"]
    maskX      = T.toDevice T.cpu $ boolMask' ["L"] paramsX
    maskY      = T.toDevice T.cpu $ boolMask' [ "idoverW", "fug", "gmoverW", "gdsoverW", "gmbsoverW"
                                              , "cgsoverW", "cgdoverW", "cgboverW", "csdoverW", "cbsoverW", "cbdoverW"
                                              ] paramsY
    neg m t    = T.add (T.negative (T.mul m t)) (T.mul (1.0 - m) t)
    !maskNeg'  = if dev == NMOS then ["cgsoverW", "cgdoverW", "cgboverW", "csdoverW", "cbsoverW", "cbdoverW"]
                                else ["id","cgsoverW", "cgdoverW", "cgboverW", "csdoverW", "cbsoverW", "cbdoverW"]
    !maskNeg   = T.toDType T.Float . T.toDevice T.cpu $ boolMask' maskNeg' paramsY
    numInputs  = length paramsX
    numOutputs = length paramsY

-- | Basic Training Run
runBase :: Args -> IO ()
runBase Args{..} = do
    putStrLn $ "Training " ++ show dev ++ " Model in " ++ show pdk ++ "."

    df' <- DF.fromFile dir

    let idoverW = T.div (df' ?? "id") (df' ?? "W")
        dfRaw   = DF.insert ["idoverW"] idoverW df'

    modelPath <- createModelDir pdk' dev'
    let tracePath = modelPath ++ "/" ++ dev' ++ "-" ++ reg' ++ ".pt"

    let dfRaw' = DF.dropNan $ DF.lookup cols dfRaw
        dfReg  = DF.rowFilter (region reg dfRaw') dfRaw'

    dfShuff    <- DF.shuffleIO dfReg

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

    let predict x = y
          where
            !i'  = T.arange 1 num' 1 . T.withDType T.Int64 $ T.defaultOpts 
            !x'  = T.indexSelect 1 i' x
            !y'  = trafo' maskY . scale' minY maxY . forward net''
                 . scale minX maxX . trafo maskX $ x'
            !w   = T.reshape [-1,1] $ T.select 1 0 x / T.select 1 0 y'
            !l   = T.reshape [-1,1] $ T.select 1 1 y'
            !v   = T.reshape [-1,1] $ T.select 1 2 y'
            !y   = T.abs $ T.cat (T.Dim 1) [ w, l, v]

    traceModel dev pdk num' paramsX paramsY predict >>= saveInferenceModel tracePath

    putStrLn $ "Final Checkpoint saved at: " ++ modelPath
    putStrLn $ "Traced Model saved at: " ++ tracePath

    mdl'' <- unTraceModel <$> T.loadScript T.WithoutRequiredGrad tracePath

    testModel reg dfRaw' paramsX' paramsY' mdl''
  where
    reg'       = if reg == 2 then "sat" else "ulp"
    pdk'       = show pdk
    dev'       = show dev
    testSplit  = 0.75
    paramsX    = ["gmoverid", "fug", "vds", "vbs"]
    paramsY    = ["idoverW", "L", "vgs"]
    paramsX'   = "id" : paramsX
    paramsY'   = ["W", "L", "vgs"]
    cols       = paramsX ++ paramsY ++ ["id", "region", "vth", "W"]
    maskX      = T.toDevice T.cpu $ boolMask' ["fug"] paramsX
    maskY      = T.toDevice T.cpu $ boolMask' ["idoverW", "L"] paramsY
    numInputs  = length paramsX
    numOutputs = length paramsY
    num'       = numInputs + 1

-- | Model Test
testModel :: Int -> DF.DataFrame T.Tensor -> [String] -> [String]
          -> (T.Tensor -> T.Tensor) -> IO ()
testModel reg df paramsX paramsY mdl = do
    df' <- DF.sampleIO 10 False $  DF.rowFilter (region reg df) df
    let x  = DF.lookup paramsX df'
        y  = DF.lookup paramsY df'
           --   DF.DataFrame paramsY 
           --  $ T.cat (T.Dim 1) [ T.abs $ (df' ?? "id") / (df' ?? "idoverW")
           --                    , T.abs $ df' ?? "L" ]
        y' = DF.DataFrame paramsY . mdl $ DF.values x
    print x
    print y
    print y'
    pure ()

-- | Experimental Model Test
testModelExp :: DF.DataFrame T.Tensor -> [String] -> [String]
          -> (T.Tensor -> T.Tensor) -> IO ()
testModelExp df paramsX paramsY mdl = do
    df' <- DF.sampleIO 10 False $ DF.rowFilter (region (-1) df) df
    let x  = DF.lookup paramsX df'
        y  = DF.lookup paramsY df'
        y' = DF.DataFrame paramsY . mdl $ DF.values x
    print x
    print y
    print y'
    pure ()

-- | Filter region
-- Experimental: -1
-- Saturation: 2
-- Sub-Threshold: 3
region :: Int -> DF.DataFrame T.Tensor -> T.Tensor
region (-1) df = T.logicalAnd m' m''
  where
    m' = T.logicalOr (T.eq (df ?? "region") 1.0)
       $ T.logicalOr (T.eq (df ?? "region") 2.0) 
                     (T.eq (df ?? "region") 3.0)
    -- m'' = T.gt (df ?? "gmoverid") 0.0
    m'' = T.logicalAnd (T.lt (df ?? "cgsoverW") 0.0)
        . T.logicalAnd (T.lt (df ?? "cgdoverW") 0.0)
        . T.logicalAnd (T.lt (df ?? "cgboverW") 0.0)
        . T.logicalAnd (T.lt (df ?? "csdoverW") 0.0)
        . T.logicalAnd (T.lt (df ?? "cbsoverW") 0.0)
        $ T.logicalAnd (T.lt (df ?? "cbdoverW") 0.0)
                       (T.gt (df ?? "gmoverid") 0.0)
region 3 df = T.logicalOr  (T.eq (df ?? "region") 3) 
            . T.logicalAnd (T.gt vds 10.0e-3)
            $ T.logicalAnd (T.gt vgs (vth - vt')) (T.lt vgs (vth + vt''))
    where
      vt    = 25.0e-3
      vt'   = 20 * vt
      vt''  = 10 * vt
      vgs   = T.abs $ df ?? "vgs"
      vds   = T.abs $ df ?? "vds"
      vth   = T.abs $ df ?? "vth"
region r df = T.eq (df ?? "region") . T.asTensor @Float $ realToFrac r
-- region r _  = error $ "Region " ++ show r ++ " not defined"
