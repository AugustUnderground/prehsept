{-# OPTIONS_GHC -Wall #-}
{-# LANGUAGE OverloadedStrings #-}


-- | Helper and Utility Functions
module Lib where

import           System.Directory
import           System.ProgressBar
import           Data.Maybe
import           Data.List
import           Data.Text.Internal.Lazy      (empty)
import           Data.Time.Clock              (getCurrentTime)
import           Data.Time.Format             (formatTime, defaultTimeLocale)
import qualified Torch                   as T

------------------------------------------------------------------------------
-- Data Types
------------------------------------------------------------------------------

-- | Available PDKs
data PDK = XH035 -- ^ X-Fab XH035 350nm Process
         | XH018 -- ^ X-Fab XH018 180nm Process
         | XT018 -- ^ X-Fab XT018 180nm Process
    deriving (Eq)

instance Show PDK where
  show XH035 = "xh035"
  show XH018 = "xh018"
  show XT018 = "xt018"

instance Read PDK where
  readsPrec _ "xh035" = [(XH035, "")]
  readsPrec _ "xh018" = [(XH018, "")]
  readsPrec _ "xt018" = [(XT018, "")]
  readsPrec _ _       = undefined

-- | Device Types
data Device = NMOS -- ^ NMOS
            | PMOS -- ^ PMOS
    deriving (Eq)

instance Show Device where
  show NMOS = "nmos"
  show PMOS = "pmos"

instance Read Device where
  readsPrec _ "nmos" = [(NMOS, "")]
  readsPrec _ "pmos" = [(PMOS, "")]
  readsPrec _ _      = undefined

------------------------------------------------------------------------------
-- General
------------------------------------------------------------------------------

-- | First of Triple
fst3 :: (a,b,c) -> a
fst3 (a,_,_) = a

-- | Create a boolean mask from a subset of column names
boolMask :: [String] -> [String] -> [Bool]
boolMask sub = map (`elem` sub)

-- | Create a boolean mask Tensor from a subset of column names
boolMask' :: [String] -> [String] -> T.Tensor
boolMask' sub set = T.asTensor' (boolMask sub set) 
                  $ T.withDType T.Bool T.defaultOpts

-- | Create Integer Index for a subset of column names
intIdx :: [String] -> [String] -> [Int]
intIdx set = fromJust . sequence . filter isJust . map (`elemIndex` set)

-- | Create Integer Index Tensor for a subset of column names
intIdx' :: [String] -> [String] -> T.Tensor
intIdx' set sub = T.asTensor' (intIdx set sub) 
                $ T.withDType T.Int64 T.defaultOpts

-- | Round Float Tensor to n digits
round :: Int -> T.Tensor -> T.Tensor
round digits = (* (1 / d)) . T.ceil . (* d)
  where
    d = T.asTensor $ 10 ** (realToFrac digits :: Float)

------------------------------------------------------------------------------
-- Saving and Loading Tensors
------------------------------------------------------------------------------

-- | Default Column Names for stored Tensors
defaultHeaders :: [String]
defaultHeaders = [ "vg", "W", "L", "temp", "M0.m1:vgs", "M0.m1:vds"
                 , "M0.m1:vbs", "M0.m1:vth", "M0.m1:vdsat", "M0.m1:gm"
                 , "M0.m1:gds", "M0.m1:gmbs", "M0.m1:betaeff", "M0.m1:cgg"
                 , "M0.m1:cgd", "M0.m1:cgs", "M0.m1:cgb", "M0.m1:cdg"
                 , "M0.m1:cdd", "M0.m1:cds", "M0.m1:cdb", "M0.m1:csg"
                 , "M0.m1:csd", "M0.m1:css", "M0.m1:csb", "M0.m1:cbg"
                 , "M0.m1:cbd", "M0.m1:cbs", "M0.m1:cbb", "M0.m1:ron"
                 , "M0.m1:id", "M0.m1:pwr", "M0.m1:gmoverid", "M0.m1:self_gain"
                 , "M0.m1:rout", "M0.m1:fug", "M0.m1:vearly", "D", "G", "S"
                 , "B" , "VB:p", "VD:p", "VG:p", "VS:p" ]

-- | Load a Pickled Tensor from file
loadTensor :: FilePath -> IO T.Tensor
loadTensor path = do
    T.IVTensor t <- T.pickleLoad path
    pure t

-- | Pickle a Tensor and Save to file
saveTensor :: T.Tensor -> FilePath -> IO ()
saveTensor t path = do
    let t' = T.IVTensor t
    T.pickleSave t' path

------------------------------------------------------------------------------
-- File System
------------------------------------------------------------------------------

-- | Current Timestamp as formatted string
currentTimeStamp :: String -> IO String
currentTimeStamp format = formatTime defaultTimeLocale format <$> getCurrentTime

-- | Current Timestamp with default formatting: "%Y%m%d-%H%M%S"
currentTimeStamp' :: IO String
currentTimeStamp' = currentTimeStamp "%Y%m%d-%H%M%S"

-- | Create a model archive directory for the given pdk and device type
createModelDir :: String -> String -> IO String
createModelDir pdk' dev' = do
    path <- (path' ++) <$> currentTimeStamp'
    createDirectoryIfMissing True path
    pure path
  where
    path' = "./models/" ++ pdk' ++ "/" ++ dev' ++ "-"

------------------------------------------------------------------------------
-- Command Line Argument Parser
------------------------------------------------------------------------------

-- | Command Line Arguments
data Args = Args { pdk  :: PDK      -- ^ PDK
                 , dev  :: Device   -- ^ NMOS | PMOS
                 , dir  :: FilePath -- ^ Path to Data
                 , num  :: Int      -- ^ Number of Epochs
                 , size :: Int      -- ^ Batch Size
                 } deriving (Show)

------------------------------------------------------------------------------
-- Progress Style
------------------------------------------------------------------------------

-- | Progress Bar Style for Training
trainStyle :: Style s
trainStyle = Style { styleOpen          = "["
                   , styleClose         = "]"
                   , styleDone          = '|'
                   , styleCurrent       = '>'
                   , styleTodo          = '-'
                   , stylePrefix        = msg "Training"
                   , stylePostfix       = msg " MSE"
                   , styleWidth         = ConstantWidth 40
                   , styleEscapeOpen    = const empty
                   , styleEscapeClose   = const empty
                   , styleEscapeDone    = const empty
                   , styleEscapeCurrent = const empty
                   , styleEscapeTodo    = const empty
                   , styleEscapePrefix  = const empty
                   , styleEscapePostfix = const empty
                   , styleOnComplete    = WriteNewline
                   }

-- | Progress Bar Style for Validation
validStyle :: Style s
validStyle = Style { styleOpen          = "["
                   , styleClose         = "]"
                   , styleDone          = '|'
                   , styleCurrent       = '>'
                   , styleTodo          = '-'
                   , stylePrefix        = msg "Validating"
                   , stylePostfix       = msg " MAE"
                   , styleWidth         = ConstantWidth 40
                   , styleEscapeOpen    = const empty
                   , styleEscapeClose   = const empty
                   , styleEscapeDone    = const empty
                   , styleEscapeCurrent = const empty
                   , styleEscapeTodo    = const empty
                   , styleEscapePrefix  = const empty
                   , styleEscapePostfix = const empty
                   , styleOnComplete    = WriteNewline
                   }
