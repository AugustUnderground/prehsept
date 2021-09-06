{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeApplications #-}

module Lib ( NetSpec (..)
           , Net
           , net
           , OP (..)
           , OPData (..)
           , getDataFromNC
           , shuffleData
           , splitData
           , xyData
           , transformData
           , scaleData
           , transformData'
           , scaleData'
           , preprocessData
           ) where

import Torch hiding (take, floor)

import Data.NetCDF
import Data.NetCDF.Vector
import Foreign.C

import Data.Maybe (mapMaybe)

import GHC.Generics
import GHC.Exts (IsList (fromList))

import System.Random.Shuffle hiding (shuffle)

-- import Control.Lens (element, (^?))
-- import Control.Monad.Cont (ContT (..), runContT)
import Control.Monad.Random hiding (fromList, split)

import Data.Kind
-- import Data.List.Split
import qualified Data.Set as S
import qualified Data.Map as M
import qualified Data.List as L
import qualified Data.Vector.Storable as SV


------------------------------------------------------------------------------
-- NEURAL NETWORK
------------------------------------------------------------------------------

data NetSpec = NetSpec { numX :: Int , numY :: Int }
    deriving (Show, Eq)

data Net = Net { l0 :: Linear
               , l1 :: Linear
               , l2 :: Linear
               , l3 :: Linear
               , l4 :: Linear
               , l5 :: Linear
               , l6 :: Linear
               , l7 :: Linear
               , l8 :: Linear
               , l9 :: Linear
               }
    deriving (Generic, Show, Parameterized)

instance Randomizable NetSpec Net where
    sample NetSpec {..} = Net <$> sample (LinearSpec numX   32)     -- Layer 0
                              <*> sample (LinearSpec 32     128)    -- Layer 1
                              <*> sample (LinearSpec 128    256)    -- Layer 2
                              <*> sample (LinearSpec 256    512)    -- Layer 3
                              <*> sample (LinearSpec 512    1024)   -- Layer 4
                              <*> sample (LinearSpec 1024   512)    -- Layer 5
                              <*> sample (LinearSpec 512    256)    -- Layer 6
                              <*> sample (LinearSpec 256    128)    -- Layer 7
                              <*> sample (LinearSpec 128    32)     -- Layer 8
                              <*> sample (LinearSpec 32     numY)   -- Layer 9

net :: Net -> Tensor -> Tensor
net Net {..} = linear l9 . relu
             . linear l8 . relu
             . linear l7 . relu
             . linear l6 . relu
             . linear l5 . relu
             . linear l4 . relu
             . linear l3 . relu
             . linear l2 . relu
             . linear l1 . relu
             . linear l0

------------------------------------------------------------------------------
-- DATA
------------------------------------------------------------------------------

data OP = OP {dev :: Device, numBatches :: Int, opData :: OPData}
    -- deriving (Eq, Ord, Show)

instance Dataset IO OP Int (Tensor, Tensor)
    where getItem OP {..} ix = pure ( toDevice dev . toDType Float . (!! ix) . inputs $ opData
                                    , toDevice dev . toDType Float . (!! ix) . outputs $ opData )
          keys OP {..} = S.fromList [ 0 .. (numBatches - 1) ]

data OPData = OPData { inputs :: [Tensor], outputs :: [Tensor] }

type SVRet a = IO (Either NcError (SV.Vector a))

getDataFromNC :: String -> [String] -> IO (M.Map String [Float])
getDataFromNC fileName params = do
    Right ncFile <- openFile fileName
    vals <- forM params 
                 (\param -> do
                    let (Just var) = ncVar ncFile param
                    Right vec <- get ncFile var :: SVRet CFloat
                    let val = map realToFrac (SV.toList vec)
                    return val)
    return (M.fromList $ zip params vals)

shuffleData :: M.Map String [Float] -> IO (M.Map String [Float])
shuffleData dataMap = do M.fromList . zip params . L.transpose
                    <$> (evalRandIO . shuffleM . L.transpose 
                                    . mapMaybe (`M.lookup` dataMap) 
                                    $ params)
    where params = M.keys dataMap

splitData :: M.Map String [Float] -> Float 
          -> (M.Map String [Float], M.Map String [Float])
splitData m s = ( M.map (take numTrainSamples) m
                     , M.map (drop numTrainSamples) m )
    where (Just numSamples) = fmap length (M.lookup (head (M.keys m)) m)
          numTrainSamples   = floor . (* s) . fromIntegral $ numSamples

xyData :: M.Map String [Float] -> [String] -> [String] -> (Tensor, Tensor)
xyData m x y = ( toTensor (mapMaybe (`M.lookup` m) x :: [[Float]])
               , toTensor (mapMaybe (`M.lookup` m) y :: [[Float]]) )
    where toTensor = transpose (Dim 0) (Dim 1) . asTensor 

-- | Applys `ln|x|` to each feature x of the given tensor. Where `x` can be
-- | masked with an [Int].
transformData :: [Int] -> Tensor -> Tensor
transformData m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.log
                  . (+ asTensor (1 :: Float))
                  . Torch.abs 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor (m :: [Int])
          t' = t * (1 - m')

-- | Applys `e^x` to each column x of the given tensor. Where `x` can be masked
-- | with [Int].
transformData' :: [Int] -> Tensor -> Tensor
transformData' m t = (+ t') . (* m') 
                  . Torch.transpose (Dim 0) (Dim 1) 
                  . Torch.exp
                  . Torch.transpose (Dim 0) (Dim 1) 
                  $ t
    where m' = asTensor (m :: [Int])
          t' = t * (1 - m')

-- | Scale data such that x ∈ [a;b]. Returns a tuple, with the scaled Tensor
-- | and a Tensor [2,num-features] with min and max values per feature for
-- | unscaling.
scaleData :: Float -> Float -> Tensor -> (Tensor, Tensor)
scaleData a b x = ( ((x - xMin) * (b' - a')) / (xMax - xMin)
                  , Torch.cat (Dim 0) [ Torch.reshape [1,-1] xMin
                                      , Torch.reshape [1,-1] xMax ])
    where (xMin, _) = Torch.minDim (Dim 0) RemoveDim x
          (xMax, _) = Torch.maxDim (Dim 0) RemoveDim x
          a' = asTensor (a :: Float)
          b' = asTensor (b :: Float)
          
-- | Un-Scale data where x ∈ [a;b] for a given min and max.
scaleData' :: Float -> Float -> Tensor -> Tensor -> Tensor
scaleData' a b s y = ((y - a') / (b' - a') * (xMax - xMin)) + xMin
    where xMin = Torch.select 0 0 s
          xMax = Torch.select 0 1 s
          a' = asTensor (a :: Float)
          b' = asTensor (b :: Float)

-- | 'preprocessData' is a convenience function to clean up main.
-- | Arguments:
-- |    computing device :: Device
-- |    lower limit scaled :: FLoat
-- |    upper limit scaled :: FLoat
-- |    transformation mask X :: [Int]
-- |    transformation mask Y :: [Int]
-- |    parameters X :: [String]
-- |    parameters Y :: [String]
-- |    train test split ratio :: Float
-- |    NetCDF Data map as obtained by `getDataFromNC`
-- |    batch size :: Int
-- |    target computing device :: Device
-- | Returns a tuple  split into train and validation sets. Additionaly scalers
-- | are returned for unscaling the data later.
-- |    -> (trainingData, validationData, scalerX, scalerY)
preprocessData :: Float -> Float -> [Int] -> [Int] -> [String] -> [String] 
                -> Float -> M.Map String [Float] -> Int -> Device
                -> (OPData, OPData, Tensor, Tensor)
preprocessData lo hi mX mY pX pY splt dat bs dev 
        = ( trainData, validData, scalerX, scalerY)
    where (rawTrain, rawValid)       = splitData dat splt
          (rawTrainX, rawTrainY)     = xyData rawTrain pX pY
          (rawValidX, rawValidY)     = xyData rawValid pX pY
          trafoTrainX                = transformData mX rawTrainX
          trafoTrainY                = transformData mY rawTrainY
          trafoValidX                = transformData mX rawValidX
          trafoValidY                = transformData mY rawValidY
          (trainX, scalerX)          = scaleData lo hi trafoTrainX
          (trainY, scalerY)          = scaleData lo hi trafoTrainY
          (validX, _)                = scaleData lo hi trafoValidX
          (validY, _)                = scaleData lo hi trafoValidY
          trainData                  = OPData (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ trainX) 
                                              (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ trainY)
          validData                  = OPData (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ validX) 
                                              (split bs (Dim 0) . toDevice dev 
                                                                . toDType Float 
                                                                $ validY)
