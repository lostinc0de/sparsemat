use crate::types::{IndexType, ValueType};
use crate::vector::Vector;

// A simple sparse vector implementation storing positions in a Vec
// Looking up values may be very inefficient
#[derive(Clone, Debug)]
pub struct SparseVec<T, I> {
    values: Vec<T>,
    indices: Vec<I>,
    dim: usize,
}

// Iterator for iterating over all values including the non-present zeroes
pub struct IterVal<T, I> {
    data: Vec<(I, T)>,
    pos: usize,
    index: usize,
}

impl<T, I> Iterator for IterVal<T, I>
where T: ValueType,
      I: IndexType {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = None;
        if self.pos < self.data.len() {
            if self.index == self.data[self.pos].0.as_usize() {
                ret = Some(self.data[self.pos].1);
                self.pos += 1;
            } else {
                ret = Some(T::zero());
            }
            self.index += 1;
        }
        ret
    }
}

impl<'a, T, I> SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    pub fn iter_sparse(&'a self) -> std::iter::Zip<std::slice::Iter<'a, I>, std::slice::Iter<'a, T>> {
        self.indices.iter().zip(self.values.iter())
    }

    pub fn n_non_zero_entries(&self) -> usize {
        self.values.len()
    }

    pub fn sort(&mut self) {
        let mut inds_vals = self.iter_sparse().map(|(&c, &v)| (c, v)).collect::<Vec<(I, T)>>();
        inds_vals.sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
        for (i, (ind, val)) in inds_vals.iter().enumerate() {
            self.indices[i] = *ind;
            self.values[i] = *val;
        }
    }

}

impl<'a, T, I> Vector<'a> for SparseVec<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    type Value = T;
    type IterVal = IterVal::<T, I>;

    fn iter(&self) -> Self::IterVal {
        let mut data = self.iter_sparse().map(|(&c, &v)| (c, v)).collect::<Vec<(I, T)>>();
        data.sort_by(|(c1, _v1), (c2, _v2)| c1.partial_cmp(c2).unwrap());
        Self::IterVal {
            data: data,
            pos: 0,
            index: 0,
        }
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec::<T>::with_capacity(cap),
            indices: Vec::<I>::with_capacity(cap),
            dim: 0,
        }
    }

    fn from_vec(vec: Vec<T>) -> Self {
        let mut indices = Vec::<I>::with_capacity(vec.len());
        for i in 0..vec.len() {
            let ind = I::as_indextype(i);
            indices.push(ind);
        }
        Self {
            values: vec,
            indices: indices,
            dim: 0,
        }
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn get(&self, i: usize) -> T {
        let pos = I::as_indextype(i);
        match self.indices.iter().enumerate().find(|(_, &x)| x == pos) {
            Some((index, _)) => self.values[index.as_usize()],
            None => T::zero(),
        }
    }

    fn get_mut(&mut self, i: usize) -> &mut T {
        if self.dim < i {
            self.dim = i;
        }
        let pos = I::as_indextype(i);
        match self.indices.iter().enumerate().find(|(_index, &x)| x == pos) {
            Some((index, _)) => &mut self.values[index.as_usize()],
            None => {
                let last = self.values.len();
                self.indices.push(pos);
                self.values.push(T::zero());
                &mut self.values[last]
            }
        }
    }

    fn add(&'a mut self, rhs: &Self) {
        for (ind, val) in rhs.iter_sparse() {
            *self.get_mut(ind.as_usize()) += *val;
        }
    }

    fn sub(&'a mut self, rhs: &Self) {
        for (ind, val) in rhs.iter_sparse() {
            *self.get_mut(ind.as_usize()) -= *val;
        }
    }

    fn scale(&'a mut self, rhs: Self::Value) {
        for x in self.values.iter_mut() {
            *x *= rhs;
        }
    }
}

impl<T, I> std::ops::AddAssign for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

impl<T, I> std::ops::SubAssign for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
    }
}

// Vector scaling
impl<T, I> std::ops::MulAssign<T> for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

impl<T, I> std::ops::Add<SparseVec<T, I>> for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret += rhs;
        ret
    }
}

impl<T, I> std::ops::Sub<SparseVec<T, I>> for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret -= rhs;
        ret
    }
}

// Vector scaling
impl<T, I> std::ops::Mul<T> for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
       let mut ret = self.clone();
       ret *= rhs;
       ret 
    }
}

// Inner product
impl<T, I> std::ops::Mul<SparseVec<T, I>> for SparseVec<T, I>
where T: ValueType,
      I: IndexType {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.inner_prod(&rhs)
    }
}
