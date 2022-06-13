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
import qualified Torch.Functional.Internal as T (where')

------------------------------------------------------------------------------
-- UTILITIES
------------------------------------------------------------------------------

-- | Main Computing Device
compDev :: T.Device
compDev = T.Device T.CUDA 1

------------------------------------------------------------------------------
-- NEURAL NETWORK
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
                                 <*> T.sample (T.LinearSpec 32   numX)

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
-- DATA PROCESSING
------------------------------------------------------------------------------

-- | Scale data to [0;1]
scale :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale  xMin xMax x  = (x - xMin) / (xMax - xMin)

-- | Un-Scale data from [0;1]
scale' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale' xMin xMax x' = x' * (xMax - xMin) + xMin

-- | Transform Masked Data 
trafo :: T.Tensor -> T.Tensor -> T.Tensor
trafo xMask x  = T.where' xMask (T.log10 . T.abs $ x) x

-- | Inverse Transform Masked Data 
trafo' :: T.Tensor -> T.Tensor -> T.Tensor
trafo' yMask y = T.where' yMask (T.pow (10.0 :: Float) y) y

------------------------------------------------------------------------------
-- TORCH SCRIPT
------------------------------------------------------------------------------

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
    T.saveParams net  (path ++ "model.pt")
    T.save (T.m1 opt) (path ++ "M1.pt")
    T.save (T.m2 opt) (path ++ "M2.pt")

-- | Trace and Return a Script Module
traceModel :: Device -> PDK -> T.Tensor -> (T.Tensor -> T.Tensor) 
           -> IO T.ScriptModule
traceModel dev pdk inputs predict = 
        T.trace (show pdk) (show dev) fun [inputs] >>= T.toScriptModule
  where
    fun = pure . map predict

-- | Save a Traced ScriptModule
saveModel :: FilePath -> T.ScriptModule -> IO ()
saveModel path model = T.saveScript model path
