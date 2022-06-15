{-# OPTIONS_GHC -Wall #-}

{-# LANGUAGE RecordWildCards #-}

-- | A module for storing Tabular Data as Tensors
module Data.Frame where

import           Lib                            hiding (round)
import           Prelude                        hiding (lookup, concat)
import qualified Torch                     as T
import qualified Torch.Functional.Internal as T        (isinf)

------------------------------------------------------------------------------
-- Base Data Type
------------------------------------------------------------------------------

-- | Data Frame
data DataFrame a = DataFrame { columns :: [String] -- ^ Unique Column Identifier
                             , values  :: a        -- ^ Data
                             } deriving (Show)

-- | Functor instance for Mapping over values
instance Functor DataFrame where
  fmap f (DataFrame c v) = DataFrame c (f v)

------------------------------------------------------------------------------
-- API
------------------------------------------------------------------------------

-- | Number of Rows in DataFrame
nRows :: DataFrame T.Tensor -> Int
nRows DataFrame{..} = head $ T.shape values

-- | Number of Columns in DataFrame
nCols :: DataFrame T.Tensor -> Int
nCols DataFrame{..} = length columns

-- | Load Tensor from file and construct DataFrame with given Header
fromFile' :: [String] -> FilePath -> IO (DataFrame T.Tensor)
fromFile' cols path = do
    vals <- loadTensor path
    pure $ DataFrame cols vals

-- | Load Tensor from file and construct DataFrame with default Header
fromFile :: FilePath -> IO (DataFrame T.Tensor)
fromFile = fromFile' defaultHeaders

-- | Look up columns
lookup :: [String] -> DataFrame T.Tensor -> DataFrame T.Tensor
lookup cols DataFrame{..} = DataFrame cols vals
  where
    idx  = intIdx columns cols
    vals = T.indexSelect' 1 idx values

-- | Shorthand for looking up a single key
(??) :: DataFrame T.Tensor -> String -> T.Tensor
(??) df key = values $ lookup [key] df

-- | Lookup Rows by index
rowSelect' :: [Int] -> DataFrame T.Tensor -> DataFrame T.Tensor
rowSelect' idx DataFrame{..} = DataFrame columns values'
  where
    values' = T.indexSelect' 0 idx values

-- | Lookup Rows by index
rowSelect :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowSelect idx DataFrame{..} = DataFrame columns values'
  where
    values' = T.indexSelect 0 idx values

-- | Filter Rows by condtion
rowFilter :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowFilter msk = rowSelect idx
  where
    idx = T.squeezeAll . T.indexSelect' 1 [0] . T.nonzero $ msk

-- | Drop given Rows from Data Frame
rowDrop :: T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
rowDrop idx df = rowSelect rows df
  where
    idx'          = T.arange 0 (nRows df) 1
                  $ T.withDType T.Int64 T.defaultOpts
    (unq, _, cnt) = T.uniqueDim 0 True False True
                  $ T.cat (T.Dim 0) [idx', idx]
    rows          = T.maskedSelect (T.lt cnt 2) unq

-- | Drop given Rows from Data Frame
rowDrop' :: [Int] -> DataFrame T.Tensor -> DataFrame T.Tensor
rowDrop' idx = rowDrop idx'
  where
    idx' = T.asTensor idx

-- | Drop all Rows with NaNs and Infs
dropNan :: DataFrame T.Tensor -> DataFrame T.Tensor
dropNan df = rowDrop idx df
  where
    infIdx = T.squeezeAll . T.indexSelect' 1 [0]
           . T.nonzero . T.isinf . values $ df
    nanIdx = T.squeezeAll . T.indexSelect' 1 [0]
           . T.nonzero . T.isnan . values $ df
    idx    = T.cat (T.Dim 0) [infIdx, nanIdx]

-- | Update given columns with new values (Tensor dimensions must match)
update :: [String] -> T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
update cols vals DataFrame{..} = DataFrame columns values'
  where
    idx     = T.asTensor' (intIdx columns cols) 
            $ T.withDType T.Int64 T.defaultOpts
    values' = T.transpose2D $ T.indexPut False [idx] 
                (T.transpose2D vals) (T.transpose2D values)

-- | Union of two data frames
union :: DataFrame T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
union  df df' = DataFrame cols vals
  where
    cols = columns df ++ columns df'
    vals = T.cat (T.Dim 1) [values df, values df']

-- | Add columns with data
insert :: [String] -> T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
insert cols vals df = df `union` DataFrame cols vals

-- | Join 2 DataFrames, columns must line up
join :: DataFrame T.Tensor -> DataFrame T.Tensor -> DataFrame T.Tensor
join df df' = DataFrame columns' values'
  where
    idx      = intIdx (columns df) (columns df')
    vals     = T.indexSelect' 1 idx (values df')
    values'  = T.cat (T.Dim 0) [values df, vals]
    columns' = columns df

-- | Concatenate a list of Data Frames
concat :: [DataFrame T.Tensor] -> DataFrame T.Tensor
concat = foldl1 join

-- | Take n Random samples from Data Frame
sampleIO :: Int -> Bool -> DataFrame T.Tensor -> IO (DataFrame T.Tensor)
sampleIO num replace df = do
    idx' <- T.toDType T.Int64 <$> T.multinomialIO idx num rep
    pure $ rowSelect idx' df
  where 
    len = nRows df
    rep = replace && (len <= num)
    idx = T.arange' 0 len 1

-- | Shuffle all rows
shuffleIO :: DataFrame T.Tensor -> IO (DataFrame T.Tensor)
shuffleIO df = sampleIO (nRows df) False df

-- | Split a dataframe according to a given ratio
trainTestSplit :: [String] -> [String] -> Float -> DataFrame T.Tensor
               -> (T.Tensor, T.Tensor, T.Tensor, T.Tensor)
trainTestSplit paramsX paramsY trainSize df = (trainX, validX, trainY, validY)
  where
    trainLen = round $     trainSize     * realToFrac (nRows df)
    validLen = round $ (1.0 - trainSize) * realToFrac (nRows df)
    trainIdx = T.arange 0               trainLen       1 
             $ T.withDType T.Int64 T.defaultOpts
    validIdx = T.arange trainLen (trainLen + validLen) 1 
             $ T.withDType T.Int64 T.defaultOpts
    trainX   = values . rowSelect trainIdx $ lookup paramsX df
    validX   = values . rowSelect validIdx $ lookup paramsX df
    trainY   = values . rowSelect trainIdx $ lookup paramsY df
    validY   = values . rowSelect validIdx $ lookup paramsY df
