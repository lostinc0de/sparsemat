use std::iter::Map;
use crate::types::{IndexType, ValueType};
use crate::vector::Vector;

// Interface for row major sparse matrix types
pub trait SparseMatrix<'a>
where Self: Sized + Clone {
    type Value: 'a + ValueType;
    type Index: 'a + IndexType;

    // Iterator for accessing values associated to row and column
    type Iter: Iterator<Item = (usize, &'a Self::Index, &'a Self::Value)>;
    type IterRow: Iterator<Item = (&'a Self::Index, &'a Self::Value)>;
    
    fn iter(&'a self) -> Self::Iter;
    fn iter_row(&'a self, row: usize) -> Self::IterRow;

    // Iterator for accessing all elements as values instead of references
    fn iter_val(&'a self) -> Map<Self::Iter, fn((usize, &'a Self::Index, &'a Self::Value)) -> (usize, usize, Self::Value)> {
        self.iter().map(|(i, &j, &val)| (i, j.as_usize(), val))
    }

    // Constant used to identify an empty entry
    const UNSET: Self::Index = Self::Index::MAX;

    // Creates a new sparse matrix with reserved space for cap non-zero entries
    // Useful for reducing allocations if the size is known
    fn with_capacity(cap: usize) -> Self;

    // Creates an empty sparse matrix
    fn new() -> Self {
        Self::with_capacity(0)
    }

    // Returns the number of rows
    fn n_rows(&self) -> usize;

    // Returns the maximum number of columns
    fn n_cols(&self) -> usize;

    // Returns the number of non-zero entries in the matrix
    fn n_non_zero_entries(&self) -> usize;

    // Returns the value at (i, j) or zero if it does not exist
    fn get(&self, i: usize, j: usize) -> Self::Value;

    // Returns the value at (i, j) as a reference
    // and adds it if the entry does not exist yet
    fn get_mut(&mut self, i: usize, j: usize) -> &mut Self::Value;

    // Adds another sparse matrix
    fn add(&'a mut self, rhs: &'a Self) {
        for i in 0..rhs.n_rows() {
            for (&col, &val) in rhs.iter_row(i) {
                let j = col.as_usize();
                *self.get_mut(i, j) += val;
            }
        }
    }

    // Subtracts another sparse matrix
    fn sub(&'a mut self, rhs: &'a Self) {
        for i in 0..rhs.n_rows() {
            for (&col, &val) in rhs.iter_row(i) {
                let j = col.as_usize();
                *self.get_mut(i, j) -= val;
            }
        }
    }

    // Performs a matrix-vector product
    fn mvp<V: Vector<'a, Value = Self::Value>>(&'a self, rhs: &V) -> V {
        let mut ret = V::with_capacity(self.n_rows());
        //self.iter().for_each(|(i, &j, &val)| ret.add_to(i, rhs.get(j.as_usize()) * val));
        //for (i, &j, &val) in self.iter() {
        //   ret.add_to(i, rhs.get(j.as_usize()) * val);
        //}
        for i in 0..self.n_rows() {
            let mut sum = Self::Value::zero();
            for (&col, &val) in self.iter_row(i) {
                let j = col.as_usize();
                sum += rhs.get(j) * val;
            }
            ret.set(i, sum);
        }
        ret
    }

    /*
    // Performs a parallel matrix-vector product
    fn mvp_parallel<V: Vector<'a, Value = Self::Value> + Send + Sync>(&'a self, rhs: &V) -> V
    where Self: Send + Sync {
        let n_threads = 4;
        let mut v = Vec::<V>::with_capacity(n_threads);
        let size = self.n_rows() / n_threads;
        for _ in 0..n_threads {
            v.push(V::with_capacity(size));
        }
        let mut handles = vec![];
        for t in 0..n_threads {
            let handle = thread::spawn(move || {
                let offset = t * size;
                for i in offset..offset + size {
                    let mut sum = Self::Value::zero();
                    for (&col, &val) in self.iter_row(i) {
                        let j = col.as_usize();
                        sum += rhs.get(j) * val;
                    }
                    v[t].set(i - offset, sum);
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let mut ret = V::with_capacity(self.n_rows());
        for t in 0..n_threads {
            let offset = t * size;
            for i in 0..size {
                ret.set(offset + i, v[t].get(i));
            }
        }
        ret
    }
    */

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
        impl<T, I> std::ops::AddAssign for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn add_assign(&mut self, rhs: Self) {
                self.add(&rhs);
            }
        }

        impl<T, I> std::ops::SubAssign for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn sub_assign(&mut self, rhs: Self) {
                self.sub(&rhs);
            }
        }
        
        // Matrix scaling
        impl<T, I> std::ops::MulAssign<T> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            fn mul_assign(&mut self, rhs: T) {
                self.scale(rhs);
            }
        }
        
        impl<T, I> std::ops::Add for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Self;
        
            fn add(self, rhs: Self) -> Self::Output {
                let mut ret = self.clone();
                ret += rhs;
                ret
            }
        }
        
        impl<T, I> std::ops::Sub for $Name<T, I>
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
        impl<T, I> std::ops::Mul<T> for $Name<T, I>
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
        impl<T, I> std::ops::Mul<Vec<T>> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = Vec<T>;
        
            fn mul(self, rhs: Vec<T>) -> Self::Output {
                self.mvp(&rhs)
            }
        }
    }
}