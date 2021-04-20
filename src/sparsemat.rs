use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Mul;
use std::cmp::PartialEq;
use std::cmp::PartialOrd;
use std::fmt::Display;
use std::fmt::Debug;

// Trait used for converting the index type to usize and vice versa
pub trait IndexType
where Self: Copy + PartialEq + AddAssign + PartialOrd + Display + Debug {
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
            // For some reason these functions are not inlined by default
            // and performance is pretty low without the hint
            #[inline]
            fn as_usize(&self) -> usize {
                *self as usize
            }
            #[inline]
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
where Self: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> + PartialEq + Display + Debug {
    fn zero() -> Self;
    fn one() -> Self;
}

impl<T> ValueType for T
where T: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> + PartialEq + Display + Debug {
    fn zero() -> Self {
        T::from(0u8)
    }
    fn one() -> Self {
        T::from(1u8)
    }
}

// Interface for sparse matrix types
pub trait SparseMat<'a>
where Self: Sized + Clone {
    type Value: 'a + ValueType;
    type Index: 'a + IndexType;

    // Iterator for accessing values associated to row and column
    type Iter: Iterator<Item = (usize, &'a Self::Index, &'a Self::Value)>;
    type IterRow: Iterator<Item = (&'a Self::Index, &'a Self::Value)>;
    
    fn iter(&'a self) -> Self::Iter;
    fn iter_row(&'a self, row: usize) -> Self::IterRow;

    //type IterVal = std::iter::Map<Self::Iter, fn((&'a Self::Index, &'a Self::Value)) -> (usize, usize, Self::Value)>;
    //fn iter_val(&'a self) -> impl Iterator<Item = (usize, usize, Self::Value)> {
    //    self.iter().map(|(i, &j, &val)| (i, j.as_usize(), val))
    //}

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
    fn add(&'a mut self, rhs: &'a Self) {
        rhs.iter().for_each(|(i, &j, &val)| *self.get_mut(i, j.as_usize()) += val);
        //for i in 0..rhs.n_rows() {
        //    for (&col, &val) in rhs.iter_row(i) {
        //        let j = col.as_usize();
        //        *self.get_mut(i, j) += val;
        //    }
        //}
    }

    // Subtracts another sparse matrix
    fn sub(&'a mut self, rhs: &'a Self) {
        rhs.iter().for_each(|(i, &j, &val)| *self.get_mut(i, j.as_usize()) -= val);
    }

    // Performs a matrix-vector product
    fn mvp(&'a self, rhs: &Vec<Self::Value>) -> Vec<Self::Value> {
        let mut ret = vec![Self::Value::zero(); self.n_rows()];
        self.iter().for_each(|(i, &j, &val)| ret[i] += rhs[j.as_usize()] * val);
        //for (i, &j, &val) in self.iter() {
        //    ret[i] += rhs[j.as_usize()] * val;
        //}
        //let mut ret = Vec::<Self::Value>::with_capacity(self.n_rows());
        //for i in 0..self.n_rows() {
        //    let mut sum = Self::Value::zero();
        //    for (&j, &val) in self.iter_row(i) {
        //        sum += rhs[j.as_usize()] * val;
        //    }
        //    ret.push(sum);
        //}
        ret
    }

    // Returns the transpose of this matrix
    fn transpose(&'a self) -> Self {
        let mut ret = Self::with_capacity(self.n_non_zero_entries());
        self.iter().for_each(|(i, &j, &val)| ret.set(j.as_usize(), i, val));
        ret
    }

    // Checks if the matrix is symmetric
    fn is_symmetric(&'a self) -> bool {
        let mut ret = true;
        for (i, &j, &val) in self.iter() {
            if self.get(j.as_usize(), i) != val {
                ret = false;
                break;
            }
        }
        ret
    }

    // Scales all values by a factor
    fn scale(&mut self, rhs: Self::Value);

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

    fn sparsity(&self) -> f64 {
        1.0f64 - self.density()
    }

    // Returns the identity matrix with dimension dim
    fn eye(dim: usize) -> Self {
        let mut ret = Self::with_capacity(dim);
        for i in 0..dim {
            ret.set(i, i, Self::Value::one());
        }
        ret
    }
}

// Since we are unable to implement a foreign trait we provide a macro
// for implementing all the basic operations for sparse matrix instantiation
macro_rules! sparsemat_ops {
    ($Name: ident) => {
        impl<T, I> AddAssign for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn add_assign(&mut self, rhs: Self) {
                self.add(&rhs);
            }
        }

        impl<T, I> SubAssign for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn sub_assign(&mut self, rhs: Self) {
                self.sub(&rhs);
            }
        }
        
        // Matrix scaling
        impl<T, I> MulAssign<T> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn mul_assign(&mut self, rhs: T) {
                self.scale(rhs);
            }
        }
        
        impl<T, I> Add for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Self;
        
            fn add(self, rhs: Self) -> Self::Output {
                let mut ret = self.clone();
                ret += rhs;
                ret
            }
        }
        
        impl<T, I> Sub for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Self;
        
            fn sub(self, rhs: Self) -> Self::Output {
                let mut ret = self.clone();
                ret -= rhs;
                ret
            }
        }
        
        // Matrix scaling
        impl<T, I> Mul<T> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Self;
        
            fn mul(self, rhs: T) -> Self::Output {
                let mut ret = self.clone();
                ret *= rhs;
                ret
            }
        }
        
        // Matrix-Vector multiplication
        impl<T, I> Mul<Vec<T>> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Vec<T>;
        
            fn mul(self, rhs: Vec<T>) -> Self::Output {
                self.mvp(&rhs)
            }
        }
    }
}
