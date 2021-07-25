use std::iter::Map;
use std::string::String;
use std::fs::File;
use std::io::Write;
use crate::types::{IndexType, ValueType};
use crate::sparsevec::SparseVec;
use crate::vector::Vector;

// Interface for row major sparse matrix types
pub trait SparseMatrix<'a>
where Self: Sized + Clone {
    type Value: 'a + ValueType;
    type Index: 'a + IndexType;

    // Constant used to identify an empty entry
    const UNSET: Self::Index = Self::Index::MAX;

    // Iterator for accessing values associated to row and column
    type IterRow: Iterator<Item = (&'a Self::Index, &'a Self::Value)>;
    type Iter: Iterator<Item = (usize, &'a Self::Index, &'a Self::Value)>;
    
    fn iter_row(&'a self, row: usize) -> Self::IterRow;
    fn iter(&'a self) -> Self::Iter;

    // Iterator for accessing all elements as values instead of references
    fn iter_val(&'a self) -> Map<Self::Iter, fn((usize, &'a Self::Index, &'a Self::Value)) -> (usize, usize, Self::Value)> {
        self.iter().map(|(i, &j, &val)| (i, j.as_usize(), val))
    }

    // Creates a new sparse matrix with reserved space for cap non-zero entries
    // Useful for reducing allocations if the size is known
    fn with_capacity(cap: usize) -> Self;

    // Creates an empty sparse matrix
    fn new() -> Self {
        Self::with_capacity(0)
    }

    // Returns the identity matrix with dimension dim
    fn eye(dim: usize) -> Self {
        let mut ret = Self::with_capacity(dim);
        for i in 0..dim {
            ret.set(i, i, Self::Value::one());
        }
        ret
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

    // Scales all values by a factor
    fn scale(&mut self, rhs: Self::Value);

    // Adds another sparse matrix
    fn add<S>(&'a mut self, rhs: &'a S)
    where S: SparseMatrix<'a, Value = Self::Value> {
        for i in 0..rhs.n_rows() {
            for (&col, &val) in rhs.iter_row(i) {
                let j = col.as_usize();
                *self.get_mut(i, j) += val;
            }
        }
    }

    // Subtracts another sparse matrix
    fn sub<S>(&'a mut self, rhs: &'a S)
    where S: SparseMatrix<'a, Value = Self::Value> {
        for i in 0..rhs.n_rows() {
            for (&col, &val) in rhs.iter_row(i) {
                let j = col.as_usize();
                *self.get_mut(i, j) -= val;
            }
        }
    }

    // Performs a matrix-vector product
    fn mvp<V>(&'a self, rhs: &V) -> V
    where V: Vector<'a, Value = Self::Value> {
        let mut ret = V::with_capacity(self.n_rows());
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

    // Returns the transpose of this matrix
    fn transpose(&'a self) -> Self {
        let mut ret = Self::with_capacity(self.n_non_zero_entries());
        for i in 0..self.n_rows() {
            for (&col, &val) in self.iter_row(i) {
                let j = col.as_usize();
                ret.set(j, i, val);
            }
        }
        ret
    }

    // Performs a product with another matrix using the column iterator
    fn prod<M>(&'a self, rhs: &'a M) -> Self
    where M: ColumnIter<'a, Index = Self::Index, Value = Self::Value> {
        if self.n_rows() != rhs.n_cols() || self.n_cols() != rhs.n_rows() {
            panic!("Dimension mismatch");
        }
        let mut ret = Self::with_capacity(self.n_non_zero_entries());
        for i in 0..self.n_rows() {
            let mut cols_vals_i = self.iter_row(i).map(|(&c, &v)| (c, v)).collect::<Vec<(Self::Index, Self::Value)>>();
            cols_vals_i.sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
            for j in 0..rhs.n_cols() {
                let mut sum = Self::Value::zero();
                rhs.iter_col(j).for_each(|(&row, &val_rhs)| {
                    cols_vals_i.iter().take_while(|(col, _v)| *col <= row).for_each(|(col, val)| {
                        if *col == row{
                            sum += *val * val_rhs;
                        }
                    })
                });
                if sum != Self::Value::zero() {
                    ret.set(i, j, sum);
                }
            }
        }
        ret
    }

    // Checks if the matrix is symmetric
    fn is_symmetric(&'a self) -> bool {
        for (i, &j, &val) in self.iter() {
            if self.get(j.as_usize(), i) != val {
                return false;
            }
        }
        true
    }

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

    // One minus the density of the matrix
    fn sparsity(&self) -> f64 {
        1.0f64 - self.density()
    }

    // Check if entries in a row are sorted by columns in ascending order
    fn is_sorted_row(&'a self, i: usize) -> bool {
        let mut prev = 0;
        for (&col, &_val) in self.iter_row(i) {
            let j = col.as_usize();
            if j < prev {
                return false;
            }
            prev = j;
        }
        true
    }

    // Check if all entries are sorted by columns in ascending order
    fn is_sorted(&'a self) -> bool {
        for i in 0..self.n_rows() {
            if !self.is_sorted_row(i) {
                return false;
            }
        }
        true
    }

    // Returns the row as a sparse vector with sorted entries
    fn get_row(&'a self, i: usize) -> SparseVec<Self::Value, Self::Index> {
        let mut ret = SparseVec::<Self::Value, Self::Index>::new();
        for (&col, &val) in self.iter_row(i) {
            let j = col.as_usize();
            ret.set(j, val);
        }
        ret.sort();
        ret
    }

    // Returns a string with all values of row i including the zeroes
    // The entries have to be sorted first, otherwise the output will be corrupted
    fn to_string_row(&'a self, i: usize) -> String {
        let mut ret = String::from("");
        let mut j = Self::Index::ZERO;
        // The entries need to be sorted
        //let mut cols_vals = self.iter_row(i).map(|(&c, &v)| (c, v)).collect::<Vec<(Self::Index, Self::Value)>>();
        //cols_vals.sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
        let row_vec = self.get_row(i);
        for (&col, &val) in row_vec.iter_sparse() {
            while j < col {
                ret += "0 ";
                j += Self::Index::ONE;
            }
            ret += &val.to_string();
            ret += " ";
            j += Self::Index::ONE;
        }
        ret
    }

    // Returns all entries of the matrix row-wise as a string
    fn to_string(&'a self) -> String {
        let mut ret = String::from("");
        for i in 0..self.n_rows() {
            ret += &self.to_string_row(i);
            ret += "\n";
        }
        ret
    }

    // Writes sparse matrix structure to a black / white BMP-file
    fn to_pbm(&'a self, filename: String) {
        let mut file = File::create(filename).unwrap();
        file.write_all(b"P1\n").unwrap();
        file.write_all(self.n_rows().to_string().as_bytes()).unwrap();
        file.write_all(b" ").unwrap();
        file.write_all(self.n_cols().to_string().as_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
        for i in 0..self.n_rows() {
            let mut j = Self::Index::ZERO;
            let mut tmp = String::from("");
            // Sort the entries first
            let mut cols = self.iter_row(i).map(|(&c, &_v)| c).collect::<Vec<Self::Index>>();
            cols.sort_by(|c1, c2| c1.partial_cmp(c2).unwrap());
            for col in cols {
                while j < col {
                    tmp += "1";
                    j += Self::Index::ONE;
                }
                tmp += "0";
                j += Self::Index::ONE;
            }
            tmp += "\n";
            file.write_all(tmp.as_bytes()).unwrap();
        }
    }
}

// Additional trait for the column iterator
// This is optional and not every sparse matrix implementation
// needs to have a column iterator
pub trait ColumnIter<'a>
where Self: SparseMatrix<'a> {
    type IterCol: Iterator<Item = (&'a Self::Index, &'a Self::Value)>;

    // Assembles the column iterator
    fn assemble_column_info(&mut self);
    // Checks if column iterator is available
    fn has_iter_col(&self) -> bool;
    // Returns the column iterator if assembled
    fn iter_col(&'a self, col: usize) -> Self::IterCol;
}

// Additional trait for sorting entries
pub trait Sortable<'a>
where Self: SparseMatrix<'a> {
    // Sorts entries of row i by columns in ascending order
    fn sort_row(&mut self, i: usize);

    // Sorts all entries of the matrix row-wise
    fn sort(&mut self) {
        for i in 0..self.n_rows() {
            self.sort_row(i);
        }
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
        impl<T, I> std::ops::Mul<DenseVec<T>> for $Name<T, I>
        where T: ValueType,
              I: IndexType {
            type Output = DenseVec<T>;
        
            fn mul(self, rhs: DenseVec<T>) -> Self::Output {
                self.mvp(&rhs)
            }
        }
    }
}
