{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE StrictData #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Neural Network Definition and Training
module Net where

import           Lib
import           GHC.Generics
import qualified Torch            as T
import qualified Torch.Extensions as T
import qualified Torch.NN         as NN

------------------------------------------------------------------------------
-- Neural Network
------------------------------------------------------------------------------

-- | Neural Network Specification
data OpNetSpec = OpNetSpec { numX   :: !Int -- ^ Number of input neurons
                           , numY   :: !Int -- ^ Number of output neurons
                           } deriving (Show, Eq)

-- | Network Architecture
data OpNet = OpNet { fc0 :: !T.Linear
                   , fc1 :: !T.Linear
                   , fc2 :: !T.Linear
                   , fc3 :: !T.Linear
                   , fc4 :: !T.Linear
                   , fc5 :: !T.Linear
                   , fc6 :: !T.Linear
                   } deriving (Generic, Show, T.Parameterized)

-- | Neural Network Weight initialization
instance T.Randomizable OpNetSpec OpNet where
    sample OpNetSpec{..} = OpNet <$> T.sample (T.LinearSpec   numX 32)
                                 <*> T.sample (T.LinearSpec   32   128)
                                 <*> T.sample (T.LinearSpec   128  512)
                                 <*> T.sample (T.LinearSpec   512  128)
                                 <*> T.sample (T.LinearSpec   128  64)
                                 <*> T.sample (T.LinearSpec   64   32)
                                 <*> T.sample (T.LinearSpec   32   numY)

-- | Neural Network Forward Pass with scaled Data
forward :: OpNet -> T.Tensor -> T.Tensor
forward OpNet{..} = T.linear fc6 . T.relu
                  . T.linear fc5 . T.relu
                  . T.linear fc4 . T.relu
                  . T.linear fc3 . T.relu
                  . T.linear fc2 . T.relu
                  . T.linear fc1 . T.relu
                  . T.linear fc0 
------------------------------------------------------------------------------
-- Serialization
------------------------------------------------------------------------------

-- | Remove Gradient for tracing / scripting
noGrad :: (NN.Parameterized f) => f -> IO f
noGrad net = do
    params <- mapM ((`T.makeIndependentWithRequiresGrad` False) . detachToCPU)
            $ NN.flattenParameters net
    pure $ NN.replaceParameters net params
  where
    detachToCPU = T.toDevice T.cpu . T.toDependent

------------------------------------------------------------------------------
-- Saving and Loading
------------------------------------------------------------------------------

-- | Save Model and Optimizer Checkpoint
saveCheckPoint :: FilePath -> OpNet -> T.Adam -> IO ()
saveCheckPoint path net opt = do
    T.saveParams net  (path ++ "/model.pt")
    T.save (T.m1 opt) (path ++ "/M1.pt")
    T.save (T.m2 opt) (path ++ "/M2.pt")

-- | Load a Saved Model and Optimizer CheckPoint
loadCheckPoint :: FilePath -> OpNetSpec -> Int -> IO (OpNet, T.Adam)
loadCheckPoint path spec iter = do
    net <- T.sample spec >>= (`T.loadParams` (path ++ "/model.pt"))
    m1' <- T.load (path ++ "/M1.pt")
    m2' <- T.load (path ++ "/M2.pt")
    let opt = T.Adam 0.9 0.999 m1' m2' iter
    pure (net, opt)

-- | Trace and Return a Script Module
traceModel :: Device -> PDK -> Int -> [String] -> [String]
           -> (T.Tensor -> T.Tensor) -> IO T.ScriptModule
traceModel dev pdk num xs ys predict = do
    !rm <- T.randnIO' [10,num] >>= T.trace name "forward" fun . singleton
    T.define rm $ "def inputs(self,x):\n\treturn " ++ show xs ++ "\n"
    T.define rm $ "def outputs(self,x):\n\treturn " ++ show ys ++ "\n"
    T.toScriptModule rm
  where
    fun   = pure . map predict
    name  = show pdk ++ "_" ++ show dev

-- | Trace to Function
unTraceModel :: T.ScriptModule -> (T.Tensor -> T.Tensor)
unTraceModel model' x = y
  where
    T.IVTensor y = T.runMethod1 model' "forward" $ T.IVTensor x

-- | Save a Traced ScriptModule
saveInferenceModel :: FilePath -> T.ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model path

-- | Load a Traced ScriptModule
loadInferenceModel :: FilePath -> IO T.ScriptModule
loadInferenceModel = T.loadScript T.WithoutRequiredGrad
