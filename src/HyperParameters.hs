{-# OPTIONS_GHC -Wall #-}

-- | Hyper Parameters of OpNet
module HyperParameters where

import qualified Torch            as T
import qualified Torch.Extensions as T

------------------------------------------------------------------------------
-- Adam
------------------------------------------------------------------------------

-- | First gradient momentum estimate, default = 0.9
β1 :: Float
β1 = 0.9
-- | second gradient momentum estimate, default = 0.999 
β2 :: Float
β2 = 0.999
-- | Learning Rate
α :: T.Tensor
α  = T.asTensor' (1.0e-3 :: Float) 
   . T.withDType T.Float . T.withDevice T.gpu $ T.defaultOpts
