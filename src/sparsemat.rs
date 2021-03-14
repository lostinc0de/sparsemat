use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Mul;
use std::cmp::PartialEq;
use std::fmt::Display;

// Trait used for converting the index type to usize and vice versa
pub trait IndexType
where Self: Copy + PartialEq + AddAssign + Display {
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    fn as_usize(&self) -> usize;
    fn as_indextype(index: usize) -> Self;
}

macro_rules! make_indextype {
    ( $t:ty ) => {
        impl IndexType for $t {
            const MAX: $t = <$t>::MAX;
            const ZERO: $t = 0 as $t;
            const ONE: $t = 1 as $t;
            fn as_usize(&self) -> usize {
                *self as usize
            }
            fn as_indextype(index: usize) -> $t {
                index as $t
            }
        }
    }
}

make_indextype!(u8);
make_indextype!(u16);
make_indextype!(u32);
make_indextype!(usize);

// Shortcut for value type trait bounds
pub trait ValueType
where Self: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> {
    fn zero() -> Self;
}

impl<T> ValueType for T
where T: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> {
    fn zero() -> Self {
        T::from(0u8)
    }
}

// Interface for sparse matrix types
pub trait SparseMat {

    type Value: ValueType;
    type Index: IndexType;

    // Constant used to identify an empty entry
    const UNSET: Self::Index = Self::Index::MAX;

    // Creates an empty sparse matrix
    fn new() -> Self;

    // Creates a new sparse matrix with reserved space for cap non-zero entries
    // Useful for reducing allocations if the size is known
    fn with_capacity(cap: usize) -> Self;

    // Returns the number of rows
    fn n_rows(&self) -> usize;

    // Returns the maximum number of columns
    fn n_cols(&self) -> usize;

    // Returns the number of non-zero entries in the matrix
    fn n_non_zero_entries(&self) -> usize;

    // Returns the value at (i, j) or zero if it does not exist
    fn get(&self, i: usize, j: usize) -> Self::Value;

    // Adds another sparse matrix
    fn add(&mut self, rhs: &Self);

    // Subtracts another sparse matrix
    fn sub(&mut self, rhs: &Self);

    // Scales all values by a factor
    fn scale(&mut self, rhs: Self::Value);

    // Performs a matrix-vector product
    fn mvp(&self, rhs: &Vec<Self::Value>) -> Vec<Self::Value>;

    // Returns the value at (i, j) as a reference
    // and adds it if the entry does not exist yet
    fn get_mut(&mut self, i: usize, j: usize) -> &mut Self::Value;

    // Sets value at (i, j) to val
    fn set(&mut self, i: usize, j: usize, val: Self::Value) {
        *self.get_mut(i, j) = val;
    }

    // Adds value to entry at (i, j)
    fn add_to(&mut self, i: usize, j: usize, val: Self::Value) {
        *self.get_mut(i, j) += val;
    }

    // Returns the density of the matrix:
    // The number of non-zero entries over the number of all entries in the matrix.
    fn density(&self) -> f64 {
        let nnz = self.n_non_zero_entries() as f64;
        let n_entries = (self.n_rows() * self.n_cols()) as f64;
        nnz / n_entries
    }
}
