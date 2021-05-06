use crate::types::{IndexType, ValueType};
use crate::vector::Vector;

// For the sparse vector an additional interface is provided
// to access the values and their positions using an iterator
pub trait SparseVector<'a>
where Self: 'a + Clone + Vector<'a> {
    type Index: 'a + IndexType;
    type Iter: Iterator<Item = (&'a Self::Index, &'a Self::Value)>;

    // Returns an iterator over all values and their positions in the vector
    fn iter(&'a self) -> Self::Iter;
}

// TODO Index list implementation?

// A simple sparse vector implementation storing positions in a Vec
// Looking up values may be very inefficient
#[derive(Clone, Debug)]
pub struct SparseVec<T, I> {
    values: Vec<T>,
    indices: Vec<I>,
}

impl<'a, T, I> Vector<'a> for SparseVec<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    type Value = T;
    type IterVal = std::slice::Iter<'a, T>;

    fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec::<T>::with_capacity(cap),
            indices: Vec::<I>::with_capacity(cap),
        }
    }

    fn get(&self, i: usize) -> T {
        let pos = I::as_indextype(i);
        match self.indices.iter().enumerate().find(|(_, &x)| x == pos) {
            Some((index, _)) => self.values[index.as_usize()],
            None => T::zero(),
        }
    }

    fn get_mut(&mut self, i: usize) -> &mut T {
        let pos = I::as_indextype(i);
        match self.indices.iter().enumerate().find(|(_, &x)| x == pos) {
            Some((index, _)) => &mut self.values[index.as_usize()],
            None => {
                let last = self.values.len();
                self.indices.push(pos);
                self.values.push(T::zero());
                &mut self.values[last]
            }
        }
    }

    fn iter_values(&'a self) -> Self::IterVal {
        self.values.iter()
    }
}

impl<'a, T, I> SparseVector<'a> for SparseVec<T, I>
where T: 'a + ValueType,
      I: 'a + IndexType {
    type Index = I;
    type Iter = std::iter::Zip<std::slice::Iter<'a, I>, std::slice::Iter<'a, T>>;

    fn iter(&'a self) -> Self::Iter {
        self.indices.iter().zip(self.values.iter())
    }
}
