import qualified Data.Array as A


data Shape = S0 () | S1 Word | S2 (Word, Word) deriving (Eq, Ord, Show)
data Index = I0 () | I1 Word | I2 (Word, Word) deriving (Eq, Ord, Show)
instance A.Ix Index where
  range (I0 (), I0 ())         = [I0 ()]
  range (I1 a, I1 b)           = map I1 (A.range (a, b))
  range (I2 (a, b), I2 (c, d)) = map I2 (A.range ((a, b), (c, d)))

  index (I0 (), I0 ())         (I0 ())     = 0
  index (I1 a, I1 b)           (I1 c)      = A.index (a, b) c
  index (I2 (a, b), I2 (c, d)) (I2 (e, f)) = A.index ((a, b), (c, d)) (e, f)

  inRange (I0 (), I0 ())         (I0 ())     = True
  inRange (I1 a, I1 b)           (I1 c)      = A.inRange (a, b) c
  inRange (I2 (a, b), I2 (c, d)) (I2 (e, f)) = A.inRange ((a, b), (c, d)) (e, f)

zeroSI :: Shape -> Index
zeroSI (S0 ())     = I0 ()
zeroSI (S1 _)      = I1 0
zeroSI (S2 (_, _)) = I2 (0, 0)

minus1SI :: Shape -> Index
minus1SI (S0 ())     = I0 ()
minus1SI (S1 a)      = I1 (a - 1)
minus1SI (S2 (a, b)) = I2 (a - 1, b - 1)

plus1SI :: Index -> Shape
plus1SI (I0 ())     = S0 ()
plus1SI (I1 a)      = S1 (a + 1)
plus1SI (I2 (a, b)) = S2 (a + 1, b + 1)

minusSIS :: Shape -> Index -> Shape
minusSIS (S0 ())     (I0 ())     = S0 ()
minusSIS (S1 a)      (I1 b)      = S1 (a - b)
minusSIS (S2 (a, b)) (I2 (c, d)) = S2 (a - c, b - d)

addIII :: Index -> Index -> Index
addIII (I0 ())     (I0 ())     = I0 ()
addIII (I1 a)      (I1 b)      = I1 (a + b)
addIII (I2 (a, b)) (I2 (c, d)) = I2 (a + c, b + d)

shapeBounds s = (zeroSI s, minus1SI s)

-- TODO: Use newtype here and instance Show to improve array printing
type Array = A.Array Index Double
array :: Shape -> [(Index, Double)] -> Array
array shape assocs = A.array (shapeBounds shape) assocs

listArray :: Shape -> [Double] -> Array
listArray shape elems = A.listArray (shapeBounds shape) elems

 -- Use this for view-like operations
ixmap :: Shape -> (Index -> Index) -> Array -> Array
ixmap s m a = A.ixmap (shapeBounds s) m a

shape :: Array -> Shape
shape arr = plus1SI upper where (_, upper) = A.bounds arr

range :: Shape -> [Index]
range s = A.range (shapeBounds s)

size :: Shape -> Word
size (S0 ())     = 0
size (S1 a)      = a
size (S2 (a, b)) = a * b

full :: Shape -> Double -> Array
full s v = listArray s (replicate (fromIntegral (size s)) v)

ones :: Shape -> Array
ones s = full s 1

zeros :: Shape -> Array
zeros s = full s 0

slice :: Array -> Index -> Shape -> Array
slice arr start stop =
  ixmap (minusSIS stop start) (addIII start) arr

concatenate :: [Array] -> Int -> Array
concatenate arrs 0 = array s (assocs 0 arrs)
  where
    addOffset a (I1 b)      = I1 (a + b)
    addOffset a (I2 (b, c)) = I2 (a + b, c)
    newOffset old (S1 a)      = old + a
    newOffset old (S2 (a, _)) = old + a
    assocs offset (arr:arrs) =
      (zip (map (addOffset offset) (range (shape arr))) (A.elems arr))
        ++ assocs (newOffset offset (shape arr)) arrs
    assocs _ [] = []
    add0 (S1 a) (S1 b) = S1 (a + b)
    add0 (S2 (a, b)) (S2 (c, _)) = S2 (a + c, b)
    s = foldr add0 (shape (head arrs)) (map shape (tail arrs))
