use crate::types::{IndexType, ValueType};
use crate::rowindexlist::*;
use crate::sparsematrix::*;
use crate::densevec::DenseVec;
use crate::sparsemat_crs::*;

// A sparse matrix implementation utilizing the row-indexlist to store values
// Appending values costs O(1) as well as iterating over entries
// For tracking entries in the index list additional space is required
// Depending on fragmentation performance maybe sufficient
// but slower than common sparse matrix storage strategies like CRS
// This struct is useful assembling a sparse matrix and can be converted to CRS afterwards
#[derive(Clone, Debug)]
pub struct SparseMatIndexList<T, I> {
    n_cols: usize,
    columns: Vec<I>,
    values: Vec<T>,
    indexlist: IndexList<I>,
    rows: Vec<I>,
    indexlist_col: IndexList<I>,
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

    // Used for copying the column info to CRS
    pub(crate) fn column_info(&self) -> (Vec<I>, IndexList<I>) {
        (self.rows.clone(), self.indexlist_col.clone())
    }

    // Creates a new sparse matrix with CRS format
    pub fn to_crs(&self) -> SparseMatCRS<T, I> {
        SparseMatCRS::from_sparsemat_index(&self)
    }
}

impl<'a, T, I> ColumnIter<'a> for SparseMatIndexList<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    type IterCol = IterCol<'a, T, I>;

    fn assemble_column_info(&mut self) {
        // Track the rows in a vec at the same positions as the columns
        self.rows.resize(self.columns.len(), Self::UNSET);
        for i in 0..self.n_rows() {
            for index in self.indexlist.iter_row(i) {
                self.rows[index] = I::as_indextype(i);
            }
        }
        // Assemble an index list with the columns instead of the rows as key
        for col in self.columns.iter() {
            let j = col.as_usize();
            self.indexlist_col.push(j);
        }
    }

    fn iter_col(&'a self, col: usize) -> Result<Self::IterCol, SparseMatError> {
        // Check if the column info for the iterator is available and consistent
        if self.rows.len() != self.columns.len() {
            return Err(SparseMatError::new("Column iterator not available - use assemble_column_info()"));
        }
        let ret = IterCol::<T, I> {
            mat: self,
            index_iter: self.indexlist_col.iter_row(col),
        };
        Ok(ret)
    }
}

impl<'a, T, I> Sortable<'a> for SparseMatIndexList<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    fn sort_row(&mut self, i: usize) {
        let mut cols_vals = self.iter_row(i).map(|(&c, &v)| (c, v)).collect::<Vec<(I, T)>>();
        cols_vals.sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
        for ((col, val), index) in cols_vals.iter().zip(self.indexlist.iter_row(i)) {
            self.columns[index] = *col;
            self.values[index] = *val;
        }
    }
}

impl<'a, T, I> SparseMatrix<'a> for SparseMatIndexList<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    type Value = T;
    type Index = I;
    type IterRow = IterRow<'a, T, I>;

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
            indexlist: IndexList::<I>::with_capacity(cap),
            rows: Vec::<I>::new(),
            indexlist_col: IndexList::<I>::new(),
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
}

pub struct IterRow<'a, T, I> {
    mat: &'a SparseMatIndexList<T, I>,
    index_iter: crate::rowindexlist::IterRow<'a, I>,
}

impl<'a, T, I> Iterator for IterRow<'a, T, I>
where I: IndexType {
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
where I: IndexType {
    type Item = (&'a I, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(index) => Some((&self.mat.rows[index], &self.mat.values[index])),
            None => None,
        }
    }
}

sparsemat_ops!(SparseMatIndexList);
