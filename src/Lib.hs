{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE FlexibleContexts #-}

module Lib ( NetSpec (..)
           , Net
           , net
           , genData
           , getDataFromNC
           , shuffleData
           , splitData
           , xyData
           ) where

import Torch hiding (take, floor)
import Data.NetCDF
import Data.NetCDF.Vector
import Foreign.C
import Data.Maybe (mapMaybe)
import GHC.Generics
import System.Random.Shuffle
import Control.Monad.Random
-- import Control.Monad.Cont (ContT (..), runContT)
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

genData :: Tensor -> Tensor
genData t = (t - 5) ^ 2 + 3

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

