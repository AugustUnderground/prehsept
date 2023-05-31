{-# OPTIONS_GHC -Wall #-}

-- | PREHSEPT
module Main where

import Lib
import Run
import Options.Applicative

main :: IO ()
main =  execParser opts >>= run
  where
    desc = "PREHSEPT"
    opts = info (args <**> helper) 
                (fullDesc <> progDesc desc 
                          <> header "Primitive Device Modeling Around the Operating Point")

args :: Parser Args
args = Args <$> option auto ( long "pdk" 
                           <> short 'k'
                           <> metavar "PDK" 
                           <> showDefault 
                           <> value XH035
                           <> help "PDK from which the data was generated" )
            <*> option auto ( long "dev" 
                           <> short 'd'
                           <> metavar "DEV" 
                           <> showDefault 
                           <> value NMOS
                           <> help "Device Type: nmos | pmos" )
            <*> strOption ( long "dir" 
                         <> short 'p'
                         <> metavar "DIR" 
                         <> showDefault 
                         <> help "Path to lookup-table as tensor" )
            <*> option auto ( long "num" 
                           <> short 'n'
                           <> metavar "NUM" 
                           <> showDefault 
                           <> value 25
                           <> help "Number of Epochs" )
            <*> option auto ( long "reg" 
                           <> short 'r'
                           <> metavar "REGION" 
                           <> showDefault 
                           <> value 2
                           <> help "Region of Operation: 2 | 3" )
            <*> option auto ( long "size" 
                           <> short 's'
                           <> metavar "SIZE" 
                           <> showDefault 
                           <> value 5000
                           <> help "Batch Size" )
            <*> switch ( long "exp" 
                      <> short 'e'
                      <> help "Experimental Mapping" )
