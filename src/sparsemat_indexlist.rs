use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use crate::row_indexlist::*;
use crate::sparsemat::*;

#[derive(Clone, Debug)]
pub struct SparseMatIndexList<T, I> {
    n_cols: usize,
    columns: Vec<I>,
    values: Vec<T>,
    index_list: RowIndexList<I>,
}

impl<T, I> SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {

    // Returns the offset for the columns and values vec
    // or UNSET if entry (i, j) does not exist
    fn find_index(&self, i: usize, j: usize) -> usize {
        let mut ret = Self::UNSET.as_usize();
        // Iterate over the index list
        for index in self.index_list.row_iter(i) {
            if self.columns[index].as_usize() == j {
                ret = index;
                break;
            }
        }
        ret
    }

    // Adds a new entry at (i, j)
    fn push(&mut self, i: usize, j: usize, val: T) -> usize {
        if j >= self.n_cols {
            self.n_cols = j + 1;
        }
        let index = self.index_list.push(i);
        self.columns.push(I::as_indextype(j));
        self.values.push(val);
        index
    }

    // Returns an iterator over all non-zero column entries for row
    pub fn row_iter_columns(&self, row: usize) -> IterColumn<I> {
        IterColumn::<I> {
            columns: &self.columns,
            index_iter: self.index_list.row_iter(row),
        }
    }

    // Returns an iterator over all non-zero values for row
    pub fn row_iter_values(&self, row: usize) -> IterValue<T, I> {
        IterValue::<T, I> {
            values: &self.values,
            index_iter: self.index_list.row_iter(row),
        }
    }
}

impl<T, I> SparseMat for SparseMatIndexList<T, I>
where T: ValueType,
      I: IndexType {

    type Value = T;
    type Index = I;

    fn new() -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::new(),
            values: Vec::<T>::new(),
            index_list: RowIndexList::<I>::new(),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_cols: 0,
            columns: Vec::<I>::with_capacity(cap),
            values: Vec::<T>::with_capacity(cap),
            index_list: RowIndexList::<I>::with_capacity(cap),
        }
    }

    fn n_rows(&self) -> usize {
        self.index_list.n_rows()
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

    fn add(&mut self, rhs: &Self) {
        for i in 0..rhs.n_rows() {
            for (j, val) in rhs.row_iter_columns(i).zip(rhs.row_iter_values(i)) {
                *self.get_mut(i, j) += *val;
            }
        }
    }

    fn sub(&mut self, rhs: &Self) {
        for i in 0..rhs.n_rows() {
            for (j, val) in rhs.row_iter_columns(i).zip(rhs.row_iter_values(i)) {
                *self.get_mut(i, j) -= *val;
            }
        }
    }

    fn scale(&mut self, rhs: T) {
        for iter in self.values.iter_mut() {
            *iter *= rhs;
        }
    }

    fn mvp(&self, rhs: &Vec<T>) -> Vec<T> {
        let mut ret = Vec::<T>::with_capacity(self.n_rows());
        for i in 0..self.n_rows() {
            let mut sum = T::zero();
            for (j, val) in self.row_iter_columns(i).zip(self.row_iter_values(i)) {
                sum += *val * rhs[j];
            }
            ret.push(sum);
        }
        ret
    }
}

pub struct IterColumn<'a, I> {
    columns: &'a Vec<I>,
    index_iter: crate::row_indexlist::Iter<'a, I>,
}

impl<'a, I> Iterator for IterColumn<'a, I>
where I: IndexType {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(index) => {
                Some(self.columns[index].as_usize())
            }
            None => None
        }
    }
}

pub struct IterValue<'a, T, I> {
    values: &'a Vec<T>,
    index_iter: crate::row_indexlist::Iter<'a, I>,
}

impl<'a, T, I> Iterator for IterValue<'a, T, I>
where I: IndexType {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.index_iter.next() {
            Some(index) => {
                Some(&self.values[index])
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
