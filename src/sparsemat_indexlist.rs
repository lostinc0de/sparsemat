use crate::types::{IndexType, ValueType};
use crate::rowindexlist::*;
use crate::sparsematrix::*;

#[derive(Clone, Debug)]
pub struct SparseMatIndexList<T, I> {
    n_cols: usize,
    columns: Vec<I>,
    values: Vec<T>,
    indexlist: RowIndexList<I>,
    rows: Vec<I>,
    indexlist_col: RowIndexList<I>,
}

impl<T, I> SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {

    // Returns the offset for the columns and values vec
    // or UNSET if entry (i, j) does not exist
    fn find_index(&self, i: usize, j: usize) -> usize {
        let col = I::as_indextype(j);
        let mut ret = Self::UNSET.as_usize();
        if i < self.n_rows() {
            // Iterate over the index list
            for index in self.indexlist.iter_row(i) {
                if self.columns[index] == col {
                    ret = index;
                    break;
                }
            }
        }
        ret
    }

    // Adds a new entry at (i, j)
    fn push(&mut self, i: usize, j: usize, val: T) -> usize {
        if j >= self.n_cols {
            self.n_cols = j + 1;
        }
        let index = self.indexlist.push(i);
        self.columns.push(I::as_indextype(j));
        self.values.push(val);
        index
    }

    pub fn assemble_column_info(&mut self) {
        self.rows.resize(self.columns.len(), Self::UNSET);
        for i in 0..self.n_rows() {
            for index in self.indexlist.iter_row(i) {
                self.rows[index] = I::as_indextype(i);
            }
        }
        for col in self.columns.iter() {
            let j = col.as_usize();
            self.indexlist_col.push(j);
        }
    }

    pub fn iter_col(&self, col: usize) -> IterCol<T, I> {
        // Check if the column info for the iterator is available and consistent
        if self.rows.len() != self.columns.len() {
            panic!("Column iterator not available - use assemble_column_info()");
        }
        IterCol::<T, I> {
            mat: self,
            index_iter: self.indexlist_col.iter_row(col),
        }
    }
}

impl<'a, T, I> SparseMatrix<'a> for SparseMatIndexList<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {

    type Value = T;
    type Index = I;
    type Iter = Iter<'a, T, I>;
    type IterRow = IterRow<'a, T, I>;

    fn iter(&self) -> Iter<T, I> {
        Iter::<T, I> {
            mat: self,
            index_iter: self.indexlist.iter(),
        }
    }

    fn iter_row(&self, row: usize) -> IterRow<T, I> {
        IterRow::<T, I> {
            mat: self,
            index_iter: self.indexlist.iter_row(row),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::with_capacity(cap),
            values: Vec::<T>::with_capacity(cap),
            indexlist: RowIndexList::<I>::with_capacity(cap),
            rows: Vec::<I>::new(),
            indexlist_col: RowIndexList::<I>::new(),
        }
    }

    fn n_rows(&self) -> usize {
        self.indexlist.n_rows()
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

    fn sort_row(&mut self, i: usize) {
        let mut cols_vals = self.iter_row(i).map(|(&c, &v)| (c, v)).collect::<Vec<(I, T)>>();
        cols_vals.as_mut_slice().sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
        for ((col, val), index) in cols_vals.iter().zip(self.indexlist.iter_row(i)) {
            self.columns[index] = *col;
            self.values[index] = *val;
        }
    }
}

pub struct IterRow<'a, T, I> {
    mat: &'a SparseMatIndexList<T, I>,
    index_iter: crate::rowindexlist::IterRow<'a, I>,
}

impl<'a, T, I> Iterator for IterRow<'a, T, I>
where T: ValueType,
      I: IndexType {
    type Item = (&'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(index) => Some((&self.mat.columns[index], &self.mat.values[index])),
            None => None,
        }
    }
}

pub struct IterCol<'a, T, I> {
    mat: &'a SparseMatIndexList<T, I>,
    index_iter: crate::rowindexlist::IterRow<'a, I>,
}

impl<'a, T, I> Iterator for IterCol<'a, T, I>
where T: ValueType,
      I: IndexType {
    type Item = (&'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(index) => Some((&self.mat.rows[index], &self.mat.values[index])),
            None => None,
        }
    }
}

pub struct Iter<'a, T, I> {
    mat: &'a SparseMatIndexList<T, I>,
    index_iter: crate::rowindexlist::Iter<'a, I>,
}

impl<'a, T, I> Iterator for Iter<'a, T, I>
where I: IndexType {
    type Item = (usize, &'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some((row, index)) => Some((row, &self.mat.columns[index], &self.mat.values[index])),
            None => None,
        }
    }
}

sparsemat_ops!(SparseMatIndexList);
