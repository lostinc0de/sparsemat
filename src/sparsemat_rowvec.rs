use crate::types::{IndexType, ValueType};
use crate::sparsematrix::*;

// A sparse matrix implementation where each row is stored in separate vec
// This implementation makes at least two allocations each row
#[derive(Clone, Debug)]
pub struct SparseMatRowVec<T, I> {
    n_cols: usize,
    nnz: usize,
    columns: Vec<Vec<I>>,
    values: Vec<Vec<T>>,
}
 
impl<T, I> SparseMatRowVec<T, I>
where T: ValueType,
      I: IndexType {

    // Returns the offset for the columns and values vec for row i
    // or UNSET if entry (i, j) does not exist
    fn find_index(&self, i: usize, j: usize) -> usize {
        let col = I::as_indextype(j);
        let mut ret = Self::UNSET.as_usize();
        if i < self.n_rows() {
            for index in 0..self.columns[i].len() {
                if self.columns[i][index] == col {
                    ret = index;
                    break;
                }
            }
        }
        ret
    }

    fn push(&mut self, i: usize, j: usize, val: T) -> usize {
        if i >= self.n_rows() {
            self.columns.resize(i + 1, Vec::<I>::new());
            self.values.resize(i + 1, Vec::<T>::new());
        }
        if j >= self.n_cols {
            self.n_cols = j + 1;
        }
        let ret = self.columns[i].len();
        self.columns[i].push(I::as_indextype(j));
        self.values[i].push(val);
        self.nnz += 1;
        ret
    }
}

impl<'a, T, I> SparseMatrix<'a> for SparseMatRowVec<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {

    type Value = T;
    type Index = I;
    type Iter = Iter<'a, T, I>;
    type IterRow = std::iter::Zip<std::slice::Iter<'a, I>, std::slice::Iter<'a, T>>;

    fn iter(&'a self) -> Self::Iter {
        Iter::<T, I> {
            mat: self,
            row: 0,
            pos: 0,
        }
    }

    fn iter_row(&'a self, row: usize) -> Self::IterRow {
        if row < self.n_rows() {
            self.columns[row].iter().zip(self.values[row].iter())
        } else {
            panic!("Invalid row {} - Max row is {}", row, self.n_rows());
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_cols: 0,
            nnz: 0,
            values: Vec::<Vec::<T>>::with_capacity(cap),
            columns: Vec::<Vec::<I>>::with_capacity(cap),
        }
    }

    fn n_rows(&self) -> usize {
        self.columns.len()
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
    
    fn n_non_zero_entries(&self) -> usize {
        self.nnz
    }

    fn get(&self, i: usize, j: usize) -> T {
        let mut ret = T::zero();
        let index = self.find_index(i, j);
        if index != Self::UNSET.as_usize() {
            ret = self.values[i][index];
        }
        ret
    }

    fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        let mut index = self.find_index(i, j);
        if index == Self::UNSET.as_usize() {
            index = self.push(i, j, T::zero());
        }        
        &mut self.values[i][index]
    }

    fn scale(&mut self, rhs: Self::Value) {
        for i in 0..self.n_rows() {
            for iter in self.values[i].iter_mut() {
                *iter *= rhs;
            }
        }
    }
}

pub struct Iter<'a, T, I> {
    mat: &'a SparseMatRowVec<T, I>,
    row: usize,
    pos: usize,
}

impl<'a, T, I> Iterator for Iter<'a, T, I>
where T: ValueType,
      I: IndexType {
    type Item = (usize, &'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.row < (self.mat.n_rows() - 1)
               && self.pos == self.mat.columns[self.row].len() {
            self.row += 1;
            self.pos = 0;
        }
        let index = self.pos;
        if index < self.mat.columns[self.row].len() {
            self.pos += 1;
            Some((self.row, &self.mat.columns[self.row][index], &self.mat.values[self.row][index]))
        } else {
            None
        }
    }
}

sparsemat_ops!(SparseMatRowVec);
