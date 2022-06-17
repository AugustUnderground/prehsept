{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Hyper Parameters of OpNet
module Torch.Extensions where

import           GHC.Float                       (float2Double)
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T (nan_to_num, powScalar')

------------------------------------------------------------------------------
-- Convenience / Syntactic Sugar
------------------------------------------------------------------------------

-- | GPU
gpu :: T.Device
gpu = T.Device T.CUDA 1

-- | CPU
cpu :: T.Device
cpu = T.Device T.CPU 0

-- | The inverse of `log10`
pow10 :: T.Tensor -> T.Tensor
pow10 = T.powScalar' 10.0

-- | Because snake_case sucks and this project uses Float instead of Double
nanToNum :: Float -> Float -> Float -> T.Tensor -> T.Tensor
nanToNum nan' posinf' neginf' self = T.nan_to_num self nan posinf neginf
  where
    nan    = float2Double nan'
    posinf = float2Double posinf'
    neginf = float2Double neginf'

-- | Default limits for `nanToNum`
nanToNum' :: T.Tensor -> T.Tensor
nanToNum' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = float2Double (2.0e32 :: Float)
    neginf = float2Double (-2.0e32 :: Float)

-- | Default limits for `nanToNum` (0.0)
nanToNum'' :: T.Tensor -> T.Tensor
nanToNum'' self = T.nan_to_num self nan posinf neginf
  where
    nan    = 0.0 :: Double
    posinf = 0.0 :: Double
    neginf = 0.0 :: Double
