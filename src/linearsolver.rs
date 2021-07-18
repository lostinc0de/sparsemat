use crate::types::*;
use crate::sparsematrix::*;
use crate::vector::*;
use crate::densevec::*;

/*
pub trait LinearSolver {
    fn solve<M, V>(mat: &M, b: &V, x: &mut V)
    where for<'a> M: SparseMatrix<'a>,
          for<'a> V: Vector<'a, Value = <M as SparseMatrix<'a>>::Value>;
}

pub struct ConjugateGradient {
}

impl LinearSolver for ConjugateGradient {
    fn solve<M, V>(mat: &M, b: &V, x: &mut V)
    where for<'a> M: SparseMatrix<'a>,
          for<'a> V: Vector<'a, Value = <M as SparseMatrix<'a>>::Value> {
        if mat.n_rows() != mat.n_cols() {
            panic!("Matrix is not symmetric");
        }
        if mat.n_rows() != b.n_entries()
            || mat.n_rows() != x.n_entries() {
            panic!("Matrix and vector size mismatch");
        }
        let len = mat.n_rows();
        // r = b - A * x
        let mut r = b.clone();
        let tmp = mat.mvp(x).clone();
        //r.sub(&tmp);
    }
}
*/

pub trait LinearSolver {
    fn solve<'a, M>(&self, mat: &'a M, b: &DenseVec<M::Value>, x: &mut DenseVec<M::Value>)
    where M: SparseMatrix<'a>,
          M::Value: FloatType;
}

pub struct ConjugateGradient {
    tol: f64,
    iter_max: usize,
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self {
            tol: 1e-12f64,
            iter_max: 10_000,
        }
    }
}

impl LinearSolver for ConjugateGradient {
    fn solve<'a, M>(&self, mat: &'a M, b: &DenseVec<M::Value>, x: &mut DenseVec<M::Value>)
    where M: SparseMatrix<'a>,
          M::Value: FloatType {
        if mat.n_rows() != mat.n_cols() {
            panic!("Matrix is not symmetric");
        }
        if mat.n_rows() != b.dim()
            || mat.n_rows() != x.dim() {
            panic!("Matrix and vector size mismatch");
        }
        // r = b - M * x
        let mut r = b.clone() - mat.mvp(x);
        let mut p = r.clone();
        let mut r_norm_squared = r.norm_squared();
        for _k in 0..self.iter_max {
            // M * p
            let mat_p = mat.mvp(&p);
            // alpha = r * r / (p * M * p)
            let alpha = r_norm_squared / p.inner_prod(&mat_p);
            // x = x + alpha * p
            *x += p.clone() * alpha;
            // r = r - alpha * (M * p)
            r -= mat_p * alpha;
            let r_norm_squared_prev = r_norm_squared;
            r_norm_squared = r.norm_squared();
            if f64::sqrt(r_norm_squared.into()) < self.tol {
                break;
            }
            // beta = r * r / (r_prev * r_prev)
            let beta = r_norm_squared / r_norm_squared_prev;
            // p = r + beta + p
            p.scale(beta);
            p.add(&r);
        }
    }
}
