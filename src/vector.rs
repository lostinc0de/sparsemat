use std::vec::Vec;
use crate::types::ValueType;

// Common interface for vectors
pub trait Vector<'a>
where Self: Sized + Clone {
    type Value: 'a + ValueType;
    type IterVal: Iterator<Item = Self::Value>;

    fn iter(&'a self) -> Self::IterVal;

    // Returns an empty vector with capacity
    fn with_capacity(cap: usize) -> Self;

    // Constructs a dense vector from an actual std::vec::Vec
    fn from_vec(vec: Vec<Self::Value>) -> Self;

    // Returns a new and empty vector
    fn new() -> Self {
        Self::with_capacity(0)
    }

    // Return the dimension of the vector
    fn dim(&self) -> usize;

    // Returns the value at position i
    fn get(&self, i: usize) -> Self::Value;

    // Returns the value at position i mutably
    // Appends entry if it does not exist yet
    fn get_mut(&mut self, i: usize) -> &mut Self::Value;

    fn add(&'a mut self, rhs: &Self);

    fn sub(&'a mut self, rhs: &Self);

    fn scale(&'a mut self, rhs: Self::Value);

    // Sets value at position i to val
    fn set(&mut self, i: usize, val: Self::Value) {
        *self.get_mut(i) = val;
    }

    // Adds value to entry at position i
    fn add_to(&mut self, i: usize, val: Self::Value) {
        *self.get_mut(i) += val;
    }

    // Computes the inner product with another vector
    fn inner_prod<V>(&'a self, rhs: &'a V) -> Self::Value
    where V: Vector<'a, Value = Self::Value> {
        self.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }

    // Computes the squared L2 norm of the vector
    fn norm_squared(&'a self) -> Self::Value {
        self.iter().map(|x| x * x).sum()
    }

    // Computes the L2 norm of the vector
    fn norm(&'a self) -> f64 {
        f64::sqrt(self.norm_squared().into())
    }
}
