use crate::types::{IndexType, ValueType};
use crate::sparsematrix::*;
use crate::sparsemat_indexlist::*;

// Implementation of a sparse matrix with compressed row storage format
#[derive(Clone, Debug)]
pub struct SparseMatCRS<T, I> {
    n_rows: usize,
    n_cols: usize,
    values: Vec<T>,
    columns: Vec<I>,
    offset_rows: Vec<I>,
}

impl<T, I> SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {

    // Create from sparse matrix with index list
    pub fn from_sparsemat_index(rhs: &SparseMatIndexList<T, I>) -> Self {
        if rhs.n_non_zero_entries() > 0 {
            let mut values = Vec::<T>::with_capacity(rhs.n_non_zero_entries());
            let mut columns = Vec::<I>::with_capacity(rhs.n_non_zero_entries());
            let mut offset_rows = Vec::<I>::with_capacity(rhs.n_rows() + 1);
            for i in 0..rhs.n_rows() {
                offset_rows.push(I::as_indextype(columns.len()));
                for (&j, &val) in rhs.iter_row(i) {
                    columns.push(j);
                    values.push(val);
                }
            }
            offset_rows.push(I::as_indextype(columns.len()));
            SparseMatCRS::<T, I> {
                n_rows: offset_rows.len() - 1,
                n_cols: rhs.n_cols(),
                values: values,
                columns: columns,
                offset_rows: offset_rows,
            }
        } else {
            SparseMatCRS::<T, I>::new()
        }
    }

    // Returns the offset for the columns and values vec
    // or UNSET if entry (i, j) does not exist
    fn find_index(&self, i: usize, j: usize) -> usize {
        let mut ret = Self::UNSET.as_usize();
        if i < self.n_rows() {
            let start = self.offset_rows[i].as_usize();
            let end = self.offset_rows[i + 1].as_usize();
            for index in start..end {
                if self.columns[index].as_usize() == j {
                    ret = index;
                    break;
                }
            }
        }
        ret
    }
    
    // Adds a value at entry (i, j) without check
    // This is very inefficient since we use insert of Vec here
    fn push(&mut self, i: usize, j: usize, val: T) -> usize {
        if j >= self.n_cols {
            self.n_cols = j + 1;
        }
        if self.offset_rows.len() == 0 {
            self.offset_rows.resize(i + 2, I::ZERO);
        } else if i >= self.n_rows() {
            let offset_last = self.offset_rows[self.offset_rows.len() - 1];
            self.offset_rows.resize(i + 2, offset_last);
            self.n_rows = i + 1;
        }
        if self.offset_rows[i] == Self::UNSET {
            panic!("Maximum number of {} entries reached", Self::UNSET);
        }
        let index = self.offset_rows[i].as_usize();
        self.columns.insert(index, I::as_indextype(j));
        self.values.insert(index, val);
        for k in (i + 1)..self.offset_rows.len() {
            self.offset_rows[k] += I::ONE;
        }
        index
    }
}

impl<'a, T, I> SparseMatrix<'a> for SparseMatCRS<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {

    type Value = T;
    type Index = I;
    type Iter = Iter<'a, T, I>;
    type IterRow = std::iter::Zip<std::slice::Iter<'a, I>, std::slice::Iter<'a, T>>;

    fn iter(&self) -> Iter::<T, I> {
        Iter::<T, I> {
            mat: self,
            row: 0,
            pos: I::ZERO,
        }
    }

    fn iter_row(&'a self, row: usize) -> Self::IterRow {
        if row < self.n_rows() {
            let start = self.offset_rows[row].as_usize();
            let end = self.offset_rows[row + 1].as_usize();
            self.columns[start..end].iter().zip(self.values[start..end].iter())
        } else {
            self.columns[0..0].iter().zip(self.values[0..0].iter())
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_rows: 0,
            n_cols: 0,
            values: Vec::<T>::with_capacity(cap),
            columns: Vec::<I>::with_capacity(cap),
            offset_rows: Vec::<I>::with_capacity(cap + 1),
        }
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_cols(&self) -> usize {
        self.n_cols
    }
    
    fn n_non_zero_entries(&self) -> usize {
        self.columns.len()
    }

    fn get(&self, i: usize, j: usize) -> T {
        let mut ret: T = T::zero();
        let index = self.find_index(i, j);
        if index != Self::UNSET.as_usize() {
            ret = self.values[index];
        }
        ret
    }

    fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        let mut index = self.find_index(i, j);
        if index == Self::UNSET.as_usize() {
            index = self.push(i, j, T::zero());
        }        
        &mut self.values[index]
    }

    fn scale(&mut self, rhs: Self::Value) {
        for iter in self.values.iter_mut() {
            *iter *= rhs;
        }
    }
}

pub struct Iter<'a, T, I> {
    mat: &'a SparseMatCRS<T, I>,
    row: usize,
    pos: I,
}

impl<'a, T, I> Iterator for Iter<'a, T, I>
where T: ValueType,
      I: IndexType {
    type Item = (usize, &'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos == self.mat.offset_rows[self.row + 1]
               && self.row < (self.mat.n_rows() - 1) {
            self.row += 1;
        }
        let index = self.pos.as_usize();
        if index < self.mat.n_non_zero_entries() {
            self.pos += I::ONE;
            Some((self.row, &self.mat.columns[index], &self.mat.values[index]))
        } else {
            None
        }
    }
}

sparsemat_ops!(SparseMatCRS);
