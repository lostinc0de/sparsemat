pub mod types;
pub mod rowindexlist;
#[macro_use]
pub mod sparsematrix;
pub mod sparsemat_indexlist;
pub mod sparsemat_crs;
pub mod sparsemat_rowvec;
pub mod sparsemat_par;
#[macro_use]
pub mod vector;
pub mod densevec;
pub mod sparsevec;
pub mod linearsolver;

pub use self::sparsematrix::*;
pub use self::sparsemat_indexlist::*;
pub use self::sparsemat_crs::*;
pub use self::sparsemat_rowvec::*;
pub use self::sparsevec::*;
pub use self::vector::*;
pub use self::densevec::*;

#[cfg(test)]
mod tests {
    use crate::sparsematrix::*;
    use crate::sparsemat_indexlist::*;
    use crate::sparsemat_crs::*;
    use crate::sparsemat_rowvec::*;
    use crate::sparsemat_par::*;
    use crate::rowindexlist::*;
    use crate::sparsevec::*;
    use crate::vector::*;
    use crate::densevec::*;
    use crate::linearsolver::*;

    #[test]
    fn check_cg<'a>() {
        let mut mat = SparseMatIndexList::<f64, u32>::new();
        mat.set(0, 0, 4.0);
        mat.set(0, 1, 1.0);
        mat.set(1, 0, 1.0);
        mat.set(1, 1, 3.0);
        let mut b = DenseVec::<f64>::new();
        b.set(0, 1.0);
        b.set(1, 2.0);
        let mut x = DenseVec::<f64>::new();
        x.set(0, 2.0);
        x.set(1, 1.0);
        let cg = ConjugateGradient::default();
        cg.solve(&mat, &b, &mut x);
        assert_eq!((x.get(0) * 10000f64).floor() / 10000f64, 0.0909);
    }

    #[test]
    fn check_sparsemat_indexlist() {
        //check_mat::<SparseMatIndexList<f32, u32>>();
        let mut sp = SparseMatIndexList::<f32, u32>::with_capacity(3);
        sp.add_to(0, 1, 4.2);
        sp.add_to(1, 2, 4.12);
        sp.add_to(2, 2, 2.12);
        sp.add_to(1, 1, 1.12);
        *sp.get_mut(1, 1) += 1.12;
        *sp.get_mut(0, 2) += 0.12;
        *sp.get_mut(0, 0) = 8.12;
        sp.set(0, 0, 7.12);
        assert_eq!(sp.get(0, 0), 7.12);
        let mut iter = sp.iter();
        assert_eq!(iter.next(), Some((0, 1, &4.2)));
        assert_eq!(iter.next(), Some((0, 2, &0.12)));
        assert_eq!(iter.next(), Some((0, 0, &7.12)));
        assert_eq!(iter.next(), Some((1, 2, &4.12)));
        let mut iter_row = sp.iter_row(2);
        assert_eq!(iter_row.next(), Some((&2, &2.12)));
        let sum = sp.clone() + sp.clone();
        assert_eq!(sum.get(0, 0), 14.24);
        let sub = sum.clone() - sp.clone();
        assert_eq!(sub.get(0, 0), sp.get(0, 0));
        let mul = sp.clone() * 2.0;
        assert_eq!(mul.get(0, 0), sum.get(0, 0));
        let v = DenseVec::from_vec(vec![2.0, 4.8, 1.2]);
        let mvp = sp.clone() * v;
        assert_eq!(mvp.get(0), 34.544);
        assert_eq!(sp.density(), 6.0 / 9.0);

        // Test column iter
        sp.assemble_column_info();
        let mut iter_col = sp.iter_col(2).unwrap();
        assert_eq!(iter_col.next(), Some((&1, &4.12)));
        assert_eq!(iter_col.next(), Some((&2, &2.12)));
        assert_eq!(iter_col.next(), Some((&0, &0.12)));
        assert_eq!(iter_col.next(), None);

        // Test sorting functionality and conversion to CRS
        let sp_crs = sp.to_crs();
        let row_str = sp.to_string_row(1);
        assert_eq!(row_str, "0 2.24 4.12 ");
        let row_str = sp_crs.to_string_row(1);
        assert_eq!(row_str, "0 2.24 4.12 ");

        // Test matrix product
        let mp = sp_crs.prod(&sp).unwrap();
        assert_eq!(mp.get(1, 2), 17.9632);

        // Add different matrix type
        sp.add(&sp_crs);
        let row_str = sp.to_string_row(1);
        assert_eq!(row_str, "0 4.48 8.24 ");

        sp.sort();
        sp.sort_row(1);
        sp.to_pbm("test.pbm".to_string());
    }

    #[test]
    fn check_sparsemat_crs() {
        let mut sp_crs = SparseMatCRS::<f32, u32>::with_capacity(3);
        sp_crs.add_to(0, 1, 4.2);
        sp_crs.add_to(2, 2, 2.12);
        sp_crs.add_to(1, 2, 4.12);
        sp_crs.add_to(3, 2, 1.12);
        sp_crs.add_to(3, 3, 5.12);
        let mut iter = sp_crs.iter();
        assert_eq!(iter.next(), Some((0, 1, &4.2)));
        assert_eq!(iter.next(), Some((1, 2, &4.12)));
        assert_eq!(iter.next(), Some((2, 2, &2.12)));
        assert_eq!(iter.next(), Some((3, 3, &5.12)));
        assert_eq!(iter.next(), Some((3, 2, &1.12)));
        assert_eq!(iter.next(), None);
        //let mut iter_all = sp_crs.iter_all();
        //assert_eq!(iter_all.next(), Some((0, 1, &4.2)));
        //assert_eq!(iter_all.next(), Some((1, 2, &4.12)));
        //assert_eq!(iter_all.next(), Some((2, 2, &2.12)));
        //assert_eq!(iter_all.next(), Some((3, 3, &5.12)));
        //assert_eq!(iter_all.next(), Some((3, 2, &1.12)));
        //assert_eq!(iter_all.next(), None);

        // Test column iter
        sp_crs.assemble_column_info();
        let mut iter_col = sp_crs.iter_col(2).unwrap();
        assert_eq!(iter_col.next(), Some((&1, &4.12)));
        assert_eq!(iter_col.next(), Some((&2, &2.12)));
        assert_eq!(iter_col.next(), Some((&3, &1.12)));
        assert_eq!(iter_col.next(), None);

        let mut iter_row = sp_crs.iter_row(0);
        assert_eq!(iter_row.next(), Some((&1, &4.2)));
        assert_eq!(iter_row.next(), None);
        let mut iter_row = sp_crs.iter_row(5);
        assert_eq!(iter_row.next(), None);
        let v = DenseVec::from_vec(vec![2.0, 4.8, 1.2, 3.4]);
        let mvp = sp_crs.clone() * v;
        assert_eq!(mvp.get(0), 20.16);
        assert_eq!(sp_crs.density(), 5.0 / 16.0);
    }

    #[test]
    fn check_sparsemat_rowvec() {
        let mut sp = SparseMatRowVec::<f32, u32>::with_capacity(3);
        sp.add_to(0, 1, 4.2);
        sp.add_to(1, 2, 4.12);
        sp.add_to(2, 2, 2.12);
        sp.add_to(1, 1, 1.12);
        *sp.get_mut(1, 1) += 1.12;
        *sp.get_mut(0, 2) += 0.12;
        *sp.get_mut(0, 0) = 8.12;
        sp.set(0, 0, 7.12);
        assert_eq!(sp.get(0, 0), 7.12);
        assert_eq!(sp.get(0, 1), 4.2);
        let mut iter = sp.iter();
        assert_eq!(iter.next(), Some((0, 1, &4.2)));
        assert_eq!(iter.next(), Some((0, 2, &0.12)));
        assert_eq!(iter.next(), Some((0, 0, &7.12)));
        assert_eq!(iter.next(), Some((1, 2, &4.12)));
        let v = DenseVec::from_vec(vec![2.0, 4.8, 1.2]);
        let mvp = sp.mvp(&v);
        assert_eq!(mvp.get(0), 34.544);
        assert_eq!(sp.density(), 6.0 / 9.0);
    }

    #[test]
    fn check_sparsemat_par() {
        let mut mat_par = SparseMatPar::<SparseMatIndexList::<f32, u32>>::with_sub_matrices(4, 16);
        mat_par.add_to(0, 1, 4.2);
        mat_par.add_to(1, 2, 4.12);
        mat_par.add_to(2, 2, 2.12);
        mat_par.add_to(1, 1, 1.12);
        *mat_par.get_mut(1, 1) += 1.12;
        *mat_par.get_mut(0, 2) += 0.12;
        *mat_par.get_mut(0, 0) = 8.12;
        mat_par.set(0, 0, 7.12);
        assert_eq!(mat_par.get(0, 0), 7.12);
        assert_eq!(mat_par.get(0, 1), 4.2);
        let mut iter = mat_par.iter();
        assert_eq!(iter.next(), Some((0, 1, &4.2)));
        assert_eq!(iter.next(), Some((0, 2, &0.12)));
        assert_eq!(iter.next(), Some((0, 0, &7.12)));
        assert_eq!(iter.next(), Some((1, 2, &4.12)));
        let v = DenseVec::from_vec(vec![2.0, 4.8, 1.2]);
        let mvp = mat_par.mvp(&v);
        assert_eq!(mvp.get(0), 34.544);
        assert_eq!(mat_par.density(), 6.0 / 9.0);
    }

    #[test]
    fn check_rowindexlist() {
        let mut list = IndexList::<u16>::new();
        list.push(1);
        list.push(1);
        list.push(2);
        list.push(4);
        list.push(1);
        assert_eq!(list.iter_row(0).next(), None);
        assert_eq!(list.n_entries(), 5);
        let mut iter = list.iter_row(1);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(4));
    }

    #[test]
    fn check_sparsevec() {
        let mut sv = SparseVec::<f64, u16>::new();
        sv.set(8, 6.0);
        sv.set(80, 6.4);
        sv.set(55, 8.2);
        sv.set(4, 4.0);
        let mut iter = sv.iter_sparse();
        assert_eq!(iter.next(), Some((&8, &6.0)));
        assert_eq!(iter.next(), Some((&80, &6.4)));
        assert_eq!(sv.get(4), 4.0);
    }
}
