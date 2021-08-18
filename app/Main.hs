{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import Data.List (foldl', intersperse)
import Data.Functor
import GHC.Generics
import Torch

data NetSpec = NetSpec { numX :: Int
                       , numY :: Int
                       }
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

genData :: Tensor -> Tensor
genData t = (t - 5) ^ 2 + 3

main :: IO ()
main = do
    initModel' <- sample $ NetSpec numX numY
    let initModel = toDevice gpu initModel'
        optim     = mkAdam 0 0.9 0.999 (flattenParameters initModel)

    trainedModel <- foldLoop initModel numIters $ \model iter -> do
        x <- randIO' [batchSize, numX]
             Data.Functor.<&> toDType Float . toDevice gpu

        let y    = genData x
            y'   = net model x
            loss = mseLoss y y'

        when (mod iter 100 == 0) $ do
            putStrLn $ "Iter: " ++ show iter ++ " | Loss: " ++ show loss

        (model', _) <- runStep model optim loss 1e-3
        return model'

    putStrLn "Done"
    return ()
  where
    gpu       = Device CUDA 0
    numX      = 4
    numY      = 4
    numIters  = 10000
    batchSize = 2000
    trainLoss = mseLoss 
    validLoss = l1Loss 
