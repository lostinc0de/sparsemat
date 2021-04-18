use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use crate::rowindexlist::*;
use crate::sparsemat::*;

#[derive(Clone, Debug)]
pub struct SparseMatIndexList<T, I> {
    n_cols: usize,
    columns: Vec<I>,
    values: Vec<T>,
    indexlist: RowIndexList<I>,
    track_cols: bool,
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
        if self.track_cols == true {
            self.rows.push(I::as_indextype(i));
            self.indexlist_col.push(j);
        }
        index
    }

    pub fn with_column_track() -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::new(),
            values: Vec::<T>::new(),
            indexlist: RowIndexList::<I>::new(),
            track_cols: true,
            rows: Vec::<I>::new(),
            indexlist_col: RowIndexList::<I>::new(),
        }
    }
}

impl<'a, T, I> SparseMat<'a> for SparseMatIndexList<T, I>
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

    fn new() -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::new(),
            values: Vec::<T>::new(),
            indexlist: RowIndexList::<I>::new(),
            track_cols: false,
            rows: Vec::<I>::new(),
            indexlist_col: RowIndexList::<I>::new(),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::with_capacity(cap),
            values: Vec::<T>::with_capacity(cap),
            indexlist: RowIndexList::<I>::with_capacity(cap),
            track_cols: false,
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
            Some(index) => {
                Some((&self.mat.columns[index], &self.mat.values[index]))
            }
            None => None
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
            Some((row, index)) => {
                Some((row, &self.mat.columns[index], &self.mat.values[index]))
            }
            None => None
        }
    }
}

impl<T, I> AddAssign for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

impl<T, I> SubAssign for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
    }
}

// Matrix scaling
impl<T, I> MulAssign<T> for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

impl<T, I> Add for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret += rhs;
        ret
    }
}

impl<T, I> Sub for SparseMatIndexList<T, I>
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
impl<T, I> Mul<T> for SparseMatIndexList<T, I>
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
impl<T, I> Mul<Vec<T>> for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Vec<T>;

    fn mul(self, rhs: Vec<T>) -> Self::Output {
        self.mvp(&rhs)
    }
}
