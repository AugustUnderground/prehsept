module Main where

import System.Environment

import Lib

-- | Map CLI Args
dispatch :: [(String, [String] -> IO ())]
dispatch = [ ("train", train)
           -- , ("serve", serve)
           ]

main :: IO ()
main = do
    (cmd:args) <- getArgs
    let (Just action) = lookup cmd dispatch
    action args
