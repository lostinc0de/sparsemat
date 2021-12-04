use std::cmp::min;
use std::cmp::max;
//use std::thread;
//use std::sync::Arc;
//use std::sync::mpsc;
//use crate::vector::Vector;
//use crate::types::{IndexType, ValueType};
use crate::sparsematrix::SparseMatrix;

// A sparse matrix implementation used for parallel operations
#[derive(Clone, Debug)]
pub struct SparseMatPar<M> {
    n_rows_sub_matrix: usize,
    n_blocks: usize,
    sub_matrices: Vec<M>,
}
 
impl<'a, M> SparseMatPar<M>
where M: SparseMatrix<'a> {
    pub fn with_sub_matrices(n_blocks: usize, max_n_rows: usize) -> Self {
        let n_rows_sub_matrix = max_n_rows / n_blocks;
        let sub_matrices = vec![M::with_capacity(n_rows_sub_matrix); n_blocks];
        Self {
            n_rows_sub_matrix: n_rows_sub_matrix,
            n_blocks: n_blocks,
            sub_matrices: sub_matrices,
        }
    }

    // Returns the index of the submatrix and the actual row index
    fn get_block_and_row_id(&self, row: usize) -> (usize, usize) {
        let block_id = min(row / self.n_rows_sub_matrix, self.n_blocks);
        let row_id = row - block_id * self.n_rows_sub_matrix;
        (block_id, row_id)
    }

    // TODO
    /*
    pub fn mvp_par<V>(&'a self, rhs: &V) -> V
    where V: Vector<'a, Value = M::Value> + Send + Sync,
          M: Send + Sync {
        let arc = Arc::new(rhs);
        let (tx, rx) = mpsc::channel();
        for b in 0..self.n_blocks {
            let rhs = arc.clone();
            let mat = &self.sub_matrices[b];
            let tx = tx.clone();
            let mut v = V::with_capacity(self.n_rows_sub_matrix);
            thread::spawn(move || {
                for i in 0..mat.n_rows() {
                    let mut sum = M::Value::zero();
                    for (&col, &val) in mat.iter_row(i) {
                        let j = col.as_usize();
                        sum += rhs.get(j) * val;
                    }
                    v.set(i, sum);
                }
                tx.send((b, v)).unwrap();
            });
        }
        let mut ret = V::with_capacity(self.n_rows());
        for _ in 0..self.n_blocks {
            let (b, v) = rx.recv().unwrap();
            let offset = b * self.n_rows_sub_matrix;
        }
        ret
    }
    */
}

impl<'a, M> SparseMatrix<'a> for SparseMatPar<M>
where M: 'a + SparseMatrix<'a> {
    type Value = M::Value;
    type Index = M::Index;
    //type Iter = Iter<'a, M>;
    type IterRow = M::IterRow;

    //fn iter(&'a self) -> Self::Iter {
    //    Iter {
    //        mat: &self,
    //        block_id: 0,
    //        iter: self.sub_matrices[0].iter(),
    //    }
    //}

    fn iter_row(&'a self, row: usize) -> Self::IterRow {
        let (block_id, row_id) = self.get_block_and_row_id(row);
        self.sub_matrices[block_id].iter_row(row_id)
    }
    
    fn with_capacity(cap: usize) -> Self {
        Self::with_sub_matrices(4, cap)
    }

    fn n_rows(&self) -> usize {
        let mut last = 0;
        // Find the last non-empty sub matrix
        for b in 0..self.sub_matrices.len() {
            if self.sub_matrices[b].empty() {
                break;
            }
            last = b;
        }
        // The number of rows equals the number of rows in the last non-empty sub matrix
        // plus the number of rows of all preceding sub matrices
        last * self.n_rows_sub_matrix + self.sub_matrices[last].n_rows()
    }

    fn n_cols(&self) -> usize {
        let mut ret = 0;
        for mat in &self.sub_matrices {
            ret = max(mat.n_cols(), ret);
        }
        ret
    }
    
    fn n_non_zero_entries(&self) -> usize {
        let mut ret = 0;
        for mat in &self.sub_matrices {
            ret += mat.n_non_zero_entries();
        }
        ret
    }

    fn get(&self, i: usize, j: usize) -> Self::Value {
        let (block_id, row_id) = self.get_block_and_row_id(i);
        self.sub_matrices[block_id].get(row_id, j)
    }

    fn get_mut(&mut self, i: usize, j: usize) -> &mut Self::Value {
        let (block_id, row_id) = self.get_block_and_row_id(i);
        self.sub_matrices[block_id].get_mut(row_id, j)
    }

    fn scale(&mut self, rhs: Self::Value) {
        for mat in &mut self.sub_matrices {
            mat.scale(rhs);
        }
    }
}
