{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FunctionalDependencies #-}

module Main where

import Control.Monad.State
import Control.Monad (when)

import Torch

model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = toCUDA $ asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias   = toCUDA $ full' [1] (3.14 :: Float)

printParams :: Linear -> IO ()
printParams params = do
    putStrLn $ "Parameters:\n\t" ++ show (toDependent $ weight params)
    putStrLn $ "Bias:\n\t" ++ show (toDependent $ bias params)

main :: IO ()
main = do
    randGen <- defaultRNG
    init' <- sample $ LinearSpec { in_features = numFeatures, out_features = 1}
    let init = toDevice gpu init'
    printParams init

    (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
        let (input, randGen') = randn [batchSize, numFeatures] 
                                      (withDevice gpu defaultOpts) 
                                      randGen

            x = toCUDA input
            s = toDevice gpu state
            (y, y') = (groundTruth x, model s x)

            -- y = toCUDA y
            -- y' = toCUDA ym

            loss = toCUDA $ mseLoss y y'

        when (i `mod` 100 == 0) $ do
            putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss

        (newParam', _) <- runStep state optimizer loss 5e-3
        let newParam = toDevice gpu newParam'

        pure (newParam, randGen')

    printParams trained
    pure ()

  where
    gpu         = Device CUDA 0
    optimizer   = Adam { beta1 = 0.9
                       , beta2 = 0.999
                       , m1    = [ zeros [1] (withDevice gpu defaultOpts)
                                 , zeros [1] (withDevice gpu defaultOpts)
                                 ]
                       , m2    = [ zeros [1] (withDevice gpu defaultOpts)
                                 , zeros [1] (withDevice gpu defaultOpts)
                                 ]
                       , iter  = 0 
                       }
    defaultRNG  = mkGenerator gpu 666
    batchSize   = 4
    numIters    = 5000
    numFeatures = 3
