use std::ops::AddAssign;
use std::ops::SubAssign;
use std::ops::MulAssign;
use std::ops::Mul;
use std::ops::Div;
use std::cmp::PartialEq;
use std::cmp::PartialOrd;
use std::iter::Sum;
use std::fmt::Display;
use std::fmt::Debug;
use std::mem::size_of;

// Trait used for converting the index type to usize and vice versa
pub trait IndexType
where Self: Copy + PartialEq + AddAssign + PartialEq + PartialOrd + Display + Debug {
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    fn as_usize(&self) -> usize;
    fn as_indextype(index: usize) -> Self;
}

macro_rules! make_indextype {
    ( $t: ty ) => {
        impl IndexType for $t {
            const MAX: $t = <$t>::MAX;
            const ZERO: $t = 0 as $t;
            const ONE: $t = 1 as $t;

            // For some reason these functions are not inlined by default
            // and performance is pretty low without the hint
            #[inline]
            fn as_usize(&self) -> usize {
                assert!(size_of::<usize>() >= size_of::<$t>(), "Index type is larger than target usize");
                *self as usize
            }

            #[inline]
            fn as_indextype(index: usize) -> $t {
                assert!(size_of::<usize>() >= size_of::<$t>(), "Index type is larger than target usize");
                index as $t
            }
        }
    }
}

make_indextype!(u8);
make_indextype!(u16);
make_indextype!(u32);
make_indextype!(u64);
make_indextype!(usize);

// Shortcut for value type trait bounds
pub trait ValueType
where Self: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> + Div<Output = Self> + PartialEq + Sum + Into<f64> + Display + Debug + Send {
    fn zero() -> Self;
    fn one() -> Self;
}

impl<T> ValueType for T
where T: Copy + From<u8> + AddAssign + SubAssign + MulAssign + Mul<Output = Self> + Div<Output = Self> + PartialEq + Sum + Into<f64> + Display + Debug + Send {
    fn zero() -> Self {
        T::from(0u8)
    }
    fn one() -> Self {
        T::from(1u8)
    }
}

pub trait FloatType {
}

impl FloatType for f32 {
}

impl FloatType for f64 {
}
