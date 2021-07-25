use crate::vector::*;
use crate::types::ValueType;

#[derive(Clone, Debug)]
pub struct DenseVec<T> {
    values: Vec<T>,
}

impl<'a, T> DenseVec<T> {
    pub fn iter_ref(&'a self) -> std::slice::Iter<'a, T> {
        self.values.iter()
    }
}

impl<'a, T> Vector<'a> for DenseVec<T>
where T: 'a + ValueType {
    type Value = T;
    type IterVal = std::iter::Cloned<std::slice::Iter<'a, T>>;

    fn iter(&'a self) -> Self::IterVal {
        self.values.iter().cloned()
    }

    fn with_capacity(cap: usize) -> Self {
        Self {
            values: Vec::<T>::with_capacity(cap),
        }
    }

    fn from_vec(vec: Vec<T>) -> Self {
        Self {
            values: vec,
        }
    }

    fn dim(&self) -> usize {
        self.values.len()
    }

    fn get(&self, i: usize) -> Self::Value {
        self.values[i]
    }

    fn get_mut(&mut self, i: usize) -> &mut Self::Value {
        if i >= self.values.len() {
            self.values.resize(i + 1, Self::Value::zero());
        }
        &mut self.values[i]
    }

    fn add(&'a mut self, rhs: &Self) {
        if self.dim() < rhs.dim() {
            panic!("Dimension mismatch");
        }
        for (x, y) in self.values.iter_mut().zip(rhs.iter()) {
            *x += y;
        }
    }

    fn sub(&'a mut self, rhs: &Self) {
        if self.dim() < rhs.dim() {
            panic!("Dimension mismatch");
        }
        for (x, y) in self.values.iter_mut().zip(rhs.iter()) {
            *x -= y;
        }
    }

    fn scale(&'a mut self, rhs: Self::Value) {
        for x in self.values.iter_mut() {
            *x *= rhs;
        }
    }
}

impl<T> std::ops::AddAssign for DenseVec<T>
where T: ValueType {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs);
    }
}

impl<T> std::ops::SubAssign for DenseVec<T>
where T: ValueType {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs);
    }
}

// Vector scaling
impl<T> std::ops::MulAssign<T> for DenseVec<T>
where T: ValueType {
    fn mul_assign(&mut self, rhs: T) {
        self.scale(rhs);
    }
}

impl<T> std::ops::Add<DenseVec<T>> for DenseVec<T>
where T: ValueType {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret += rhs;
        ret
    }
}

impl<T> std::ops::Sub<DenseVec<T>> for DenseVec<T>
where T: ValueType {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        ret -= rhs;
        ret
    }
}

// Vector scaling
impl<T> std::ops::Mul<T> for DenseVec<T>
where T: ValueType {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
       let mut ret = self.clone();
       ret *= rhs;
       ret 
    }
}

// Inner product
impl<T> std::ops::Mul<DenseVec<T>> for DenseVec<T>
where T: ValueType {
    type Output = T;

    fn mul(self, rhs: Self) -> Self::Output {
        self.inner_prod(&rhs)
    }
}
