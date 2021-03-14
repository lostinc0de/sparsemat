use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Add;
use std::ops::Sub;
use std::ops::Mul;
use std::slice::Iter;
use crate::sparsemat::*;
use crate::sparsemat_indexlist::*;

// Implementation of a sparse matrix with compressed row storage format
#[derive(Clone, Debug)]
pub struct SparseMatCRS<T, I> {
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
        let mut values = Vec::<T>::with_capacity(rhs.n_non_zero_entries());
        let mut columns = Vec::<I>::with_capacity(rhs.n_non_zero_entries());
        let mut offset_rows = Vec::<I>::with_capacity(rhs.n_rows() + 1);
        for i in 0..rhs.n_rows() {
            offset_rows.push(I::as_indextype(columns.len()));
            columns.extend(rhs.row_iter_columns(i).map(|x| I::as_indextype(x)));
            values.extend(rhs.row_iter_values(i));
        }
        offset_rows.push(I::as_indextype(columns.len()));
        SparseMatCRS::<T, I> {
            n_cols: rhs.n_cols(),
            values: values,
            columns: columns,
            offset_rows: offset_rows,
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
        }
        let index = self.offset_rows[i];
        if index == Self::UNSET {
            panic!("Maximum number of {} entries reached", Self::UNSET);
        }
        let index = index.as_usize();
        self.columns.insert(index, I::as_indextype(j));
        self.values.insert(index, val);
        for k in (i + 1)..self.offset_rows.len() {
            self.offset_rows[k] += I::ONE;
        }
        index
    }

    // Returns an iterator over all non-zero column indices for row
    pub fn row_iter_columns(&self, row: usize) -> IterColumn<I> {
        IterColumn::<I> {
            columns: &self.columns,
            pos: self.offset_rows[row].as_usize(),
            end: self.offset_rows[row + 1].as_usize(),
        }
    }

    // Returns an iterator over all non-zero values for row
    pub fn row_iter_values(&self, row: usize) -> Iter<T> {
        let start = self.offset_rows[row].as_usize();
        let end = self.offset_rows[row + 1].as_usize();
        self.values.as_slice()[start..end].iter()
    }
}

impl<T, I> SparseMat for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {

    type Value = T;
    type Index = I;

    fn new() -> Self {
        Self {
            n_cols: 0,
            values: Vec::<T>::new(),
            columns: Vec::<I>::new(),
            offset_rows: Vec::<I>::new(),
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            n_cols: 0,
            values: Vec::<T>::with_capacity(cap),
            columns: Vec::<I>::with_capacity(cap),
            offset_rows: Vec::<I>::with_capacity(cap + 1),
        }
    }

    fn n_rows(&self) -> usize {
        let mut ret = self.offset_rows.len();
        if ret > 0 {
            ret -= 1;
        }
        ret
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
    pos: usize,
    end: usize,
}

impl<'a, I> Iterator for IterColumn<'a, I>
where I: IndexType {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.pos < self.end {
            let pos_tmp = self.pos;
            self.pos += 1;
            Some(self.columns[pos_tmp].as_usize())
        } else {
            None
        };
        ret
    }
}

impl<T, I> AddAssign for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

impl<T, I> SubAssign for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
    }
}

impl<T, I> MulAssign<T> for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

impl<T, I> Add for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret += rhs;
        ret
    }
}

impl<T, I> Sub for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret -= rhs;
        ret
    }
}

impl<T, I> Mul<T> for SparseMatCRS<T, I>
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
impl<T, I> Mul<Vec<T>> for SparseMatCRS<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Vec<T>;

    fn mul(self, rhs: Vec<T>) -> Self::Output {
        self.mvp(&rhs)
    }
}
