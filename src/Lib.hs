{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE StrictData #-}
{-# LANGUAGE OverloadedStrings #-}


-- | Helper and Utility Functions
module Lib where

import           Data.Maybe
import           Data.List
import           Data.Text.Internal.Lazy        (empty)
import           Data.Text.Lazy                 (pack)
import           Data.Time.Clock                (getCurrentTime)
import           Data.Time.Format               (formatTime, defaultTimeLocale)
import           System.Directory
import           System.ProgressBar
import qualified Torch                     as T
import qualified Torch.Extensions          as T
import qualified Torch.Functional.Internal as T (where')

------------------------------------------------------------------------------
-- Data Types
------------------------------------------------------------------------------

-- | Available PDKs
data PDK = XH035    -- ^ X-Fab XH035 350nm Process
         | XH018    -- ^ X-Fab XH018 180nm Process
         | XT018    -- ^ X-Fab XT018 180nm Process
         | GPDK180  -- ^ Fictional Cadence GPDK180 GPDK 180nm Process
         | GPDK090  -- ^ Fictional Cadence GPDK090 GPDK 90nm Process
         | GPDK045  -- ^ Fictional Cadence GPDK045 GPDK 45nm Process
         | SKY130   -- ^ SkyWater SKY130 130nm Process
         | PTM130   -- ^ Fictional Predictive Technology Models PTM130 130nm Process
    deriving (Eq)

instance Show PDK where
  show XH035   = "xh035"
  show XH018   = "xh018"
  show XT018   = "xt018"
  show GPDK180 = "gpdk180"
  show GPDK090 = "gpdk090"
  show GPDK045 = "gpdk045"
  show SKY130  = "sky130"
  show PTM130  = "ptm130"

instance Read PDK where
  readsPrec _ "xh035"   = [(XH035, "")]
  readsPrec _ "xh018"   = [(XH018, "")]
  readsPrec _ "xt018"   = [(XT018, "")]
  readsPrec _ "gpdk180" = [(GPDK180, "")]
  readsPrec _ "gpdk090" = [(GPDK090, "")]
  readsPrec _ "gpdk045" = [(GPDK045, "")]
  readsPrec _ "sky130"  = [(SKY130, "")]
  readsPrec _ "ptm130"  = [(PTM130, "")]
  readsPrec _ _         = undefined

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

-- | Uncurry triple
uncurry' :: (a -> b -> c -> d) -> (a, b, c) -> d
uncurry' f (a,b,c) = f a b c

-- | Create single element list from given element
singleton :: a -> [a]
singleton x = [x]

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
-- Scaling and Transforming
------------------------------------------------------------------------------

-- | Scale data to [0;1]
scale :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale xMin xMax x = (x - xMin) / (xMax - xMin)

-- | Un-Scale data from [0;1]
scale' :: T.Tensor -> T.Tensor -> T.Tensor -> T.Tensor
scale' xMin xMax x = (x * (xMax - xMin)) + xMin

-- | Apply log10 to masked data
trafo :: T.Tensor -> T.Tensor -> T.Tensor
trafo xMask x = T.where' xMask (T.log10 $ T.abs x) x

-- | Apply pow10 to masked data
trafo' :: T.Tensor -> T.Tensor -> T.Tensor
trafo' xMask x = T.where' xMask (T.pow10 x) x

------------------------------------------------------------------------------
-- Saving and Loading Tensors
------------------------------------------------------------------------------

-- | Default Column Names for stored Tensors
columnHeader :: [String]
columnHeader = [ "W", "L", "M", "temp", "region"
               , "vgs", "vds", "vbs", "vth", "vdsat", "vearly"
               , "gm", "gds", "gmbs", "self_gain", "ron", "rout"
               , "id", "gmoverid", "fug", "betaeff", "pwr"
               , "cgg", "cgd", "cgs", "cgb"
               , "cdg", "cdd", "cds", "cdb"
               , "csg", "csd", "css", "csb"
               , "cbg", "cbd", "cbs", "cbb" ]

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

-- | Just for the notebooks, to create a directory above
createModelDir' :: String -> String -> IO String
createModelDir' pdk' dev' = do
    path <- (path' ++) <$> currentTimeStamp'
    createDirectoryIfMissing True path
    pure path
  where
    path' = "../models/" ++ pdk' ++ "/" ++ dev' ++ "-"

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
trainStyle :: Int -> Style s
trainStyle epoch = Style { styleOpen          = "╢"
                         , styleClose         = "╟"
                         , styleDone          = '█'
                         , styleCurrent       = '░'
                         , styleTodo          = '░'
                         , stylePrefix        = msg . pack $ "Training Epoch " 
                                                                ++ show epoch
                         , stylePostfix       = percentage
                         , styleWidth         = ConstantWidth 60
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
validStyle :: Int -> Style s
validStyle epoch = Style { styleOpen          = "╢"
                         , styleClose         = "╟"
                         , styleDone          = '█'
                         , styleCurrent       = '░'
                         , styleTodo          = '░'
                         , stylePrefix        = msg . pack $ "Validation Epoch " 
                                                                ++ show epoch
                         , stylePostfix       = percentage
                         , styleWidth         = ConstantWidth 60
                         , styleEscapeOpen    = const empty
                         , styleEscapeClose   = const empty
                         , styleEscapeDone    = const empty
                         , styleEscapeCurrent = const empty
                         , styleEscapeTodo    = const empty
                         , styleEscapePrefix  = const empty
                         , styleEscapePostfix = const empty
                         , styleOnComplete    = WriteNewline
                         }
