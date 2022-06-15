{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Neural Network Definition and Training
module Net where

import           Lib
import           GHC.Generics
import qualified Torch                     as T
import qualified Torch.NN                  as NN
import qualified Torch.Functional.Internal as T (where')

------------------------------------------------------------------------------
-- Utilities
------------------------------------------------------------------------------

-- | GPU
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU
cpu :: T.Device
cpu = T.Device T.CPU 0

------------------------------------------------------------------------------
-- Neural Network
------------------------------------------------------------------------------

-- | Neural Network Specification
data OpNetSpec = OpNetSpec { numX :: Int -- ^ Number of input neurons
                           , numY :: Int -- ^ Number of output neurons
                           } deriving (Show, Eq)

-- | Network Architecture
data OpNet = OpNet { fc0 :: T.Linear
                   , fc1 :: T.Linear
                   , fc2 :: T.Linear
                   , fc3 :: T.Linear
                   , fc4 :: T.Linear
                   , fc5 :: T.Linear
                   , fc6 :: T.Linear
                   } deriving (Generic, Show, T.Parameterized)

-- | Neural Network Weight initialization
instance T.Randomizable OpNetSpec OpNet where
    sample OpNetSpec{..} = OpNet <$> T.sample (T.LinearSpec numX 32)
                                 <*> T.sample (T.LinearSpec 32   128)
                                 <*> T.sample (T.LinearSpec 128  512)
                                 <*> T.sample (T.LinearSpec 512  128)
                                 <*> T.sample (T.LinearSpec 128  64)
                                 <*> T.sample (T.LinearSpec 64   32)
                                 <*> T.sample (T.LinearSpec 32   numY)

-- | Neural Network Forward Pass
forward :: OpNet -> T.Tensor -> T.Tensor
forward OpNet{..} = T.linear fc6 . T.relu
                  . T.linear fc5 . T.relu
                  . T.linear fc4 . T.relu
                  . T.linear fc3 . T.relu
                  . T.linear fc2 . T.relu
                  . T.linear fc1 . T.relu
                  . T.linear fc0

------------------------------------------------------------------------------
-- Data Processing
------------------------------------------------------------------------------

-- | Scale data to [0;1]
scale :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale  xMin xMax x  = (x - xMin) / (xMax - xMin)

-- | Un-Scale data from [0;1]
scale' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale' xMin xMax x' = x' * (xMax - xMin) + xMin

-- | Transform Masked Data 
trafo :: T.Tensor -> T.Tensor -> T.Tensor
trafo xMask x  = T.where' cond (T.log10 . T.abs $ x) x
  where
    cond =  T.logicalAnd xMask (x `T.ne` 0.0)

-- | Inverse Transform Masked Data 
trafo' :: T.Tensor -> T.Tensor -> T.Tensor
trafo' yMask y = T.where' yMask (T.pow (10.0 :: Float) y) y

------------------------------------------------------------------------------
-- Serialization
------------------------------------------------------------------------------

-- | Remove Gradient for tracing / scripting
noGrad :: (NN.Parameterized f) => f -> IO f
noGrad net = do
    params  <- mapM (T.detach . T.toDependent) $ NN.flattenParameters net
    params' <- mapM (`T.makeIndependentWithRequiresGrad` False) params
    pure $ NN.replaceParameters net params'

-- | Wrapper for Network predictions including scaling and transformation
predictor :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor 
          -> T.Tensor -> OpNet -> T.Tensor -> T.Tensor
predictor xMin xMax yMin yMax xMask yMask net = trafo' yMask     -- Reverse Transform Outputs
                                              . scale' yMin yMax -- Unscale NN Outputs
                                              . forward net      -- Feed through NN
                                              . scale xMin xMax  -- Scale NN Inputs
                                              . trafo xMask      -- Transform NN Inputs

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
loadCheckPoint :: FilePath -> Int -> Int -> Int -> IO (OpNet, T.Adam)
loadCheckPoint path numX numY iter = do
    net <- T.sample (OpNetSpec numX numY) 
                >>= (`T.loadParams` (path ++ "/model.pt"))
    m1' <- T.load (path ++ "/M1.pt")
    m2' <- T.load (path ++ "/M2.pt")
    let opt = T.Adam 0.9 0.999 m1' m2' iter
    pure (net, opt)

-- | Trace and Return a Script Module
traceModel :: Device -> PDK -> T.Tensor -> (T.Tensor -> T.Tensor) 
           -> IO T.ScriptModule
traceModel dev pdk inputs predict = 
        T.trace name "forward" fun [inputs] >>= T.toScriptModule
  where
    fun = mapM (T.detach . predict)
    name = show pdk ++ "_" ++ show dev

-- | Save a Traced ScriptModule
saveInferenceModel :: FilePath -> T.ScriptModule -> IO ()
saveInferenceModel path model = T.saveScript model path

-- | Load a Traced ScriptModule
loadInferenceModel :: FilePath -> IO T.ScriptModule
loadInferenceModel = T.loadScript T.WithoutRequiredGrad
