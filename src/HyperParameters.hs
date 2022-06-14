{-# OPTIONS_GHC -Wall #-}

-- | Hyper Parameters of OpNet
module HyperParameters where

import           Net
import qualified Torch as T

------------------------------------------------------------------------------
-- Adam
------------------------------------------------------------------------------

-- | First gradient momentum estimate, default = 0.9
β1 :: Float
β1 = 0.5
-- | second gradient momentum estimate, default = 0.999 
β2 :: Float
β2 = 0.93
-- | Learning Rate
α :: T.Tensor
α  = T.asTensor' (0.001 :: Float) 
   . T.withDType T.Float . T.withDevice gpu $ T.defaultOpts
