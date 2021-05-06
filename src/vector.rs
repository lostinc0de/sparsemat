use std::vec::Vec;
use crate::types::ValueType;

// Common interface for vectors
pub trait Vector<'a>
where Self: Sized {
    type Value: 'a + ValueType;
    type IterVal: Iterator<Item = &'a Self::Value>;

    // Returns an empty vector with capacity
    fn with_capacity(cap: usize) -> Self;

    // Returns a new and empty vector
    fn new() -> Self {
        Self::with_capacity(0)
    }

    // Returns the value at position i
    fn get(&self, i: usize) -> Self::Value;

    // Returns the value at position i mutably
    // Appends entry if it does not exist yet
    fn get_mut(&mut self, i: usize) -> &mut Self::Value;

    // Sets value at position i to val
    fn set(&mut self, i: usize, val: Self::Value) {
        *self.get_mut(i) = val;
    }

    // Adds value to entry at position i
    fn add_to(&mut self, i: usize, val: Self::Value) {
        *self.get_mut(i) += val;
    }

    fn iter_values(&'a self) -> Self::IterVal;
}

// Implement the Vector interface for std::vec::Vec
impl<'a, T> Vector<'a> for Vec<T>
where T: 'a + ValueType {
    type Value = T;
    type IterVal = std::slice::Iter<'a, T>;

    fn with_capacity(cap: usize) -> Self {
        Vec::<T>::with_capacity(cap)
    }

    fn get(&self, i: usize) -> Self::Value {
        self[i]
    }

    fn get_mut(&mut self, i: usize) -> &mut Self::Value {
        if i >= self.len() {
            self.resize(i + 1, Self::Value::zero());
        }
        &mut self[i]
    }

    fn iter_values(&'a self) -> Self::IterVal {
        self.iter()
    }
}
