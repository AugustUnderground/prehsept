{-# OPTIONS_GHC -Wall #-}

-- | Hyper Parameters of OpNet
module Torch.Extensions where

import qualified Torch as T

-- | `pow` the other way around
pow' :: Float -> T.Tensor -> T.Tensor
pow' base = T.powt base'
  where
    base' = T.asTensor ([base] :: [Float])

-- | The inverse of `log10`
pow10 :: T.Tensor -> T.Tensor
pow10 = pow' 10.0
