#[macro_use]
pub mod sparsemat;
pub mod sparsemat_indexlist;
pub mod sparsemat_crs;
pub mod sparsemat_rowvec;
pub mod rowindexlist;

#[cfg(test)]
mod tests {
    use crate::sparsemat::*;
    use crate::sparsemat_indexlist::*;
    use crate::sparsemat_crs::*;
    use crate::sparsemat_rowvec::*;
    use crate::rowindexlist::*;

    /*
    fn check_mat<'a, T, I, M: SparseMat<'a, Value = T, Index = I>>()
    where T: ValueType,
          I: IndexType {
        let mut mat = M::new();
        mat.add_to(0, 1, 4.2);
        mat.add_to(1, 2, 4.12);
        mat.add_to(2, 2, 2.12);
        mat.add_to(1, 1, 1.12);
        *mat.get_mut(1, 1) += 1.12;
        *mat.get_mut(0, 2) += 0.12;
        *mat.get_mut(0, 0) = 8.12;
        mat.set(0, 0, 7.12);
        assert_eq!(mat.get(0, 0), 7.12);
        let mut iter = mat.iter();
        assert_eq!(iter.next(), Some((0, &1, &4.2)));
        assert_eq!(iter.next(), Some((0, &2, &0.12)));
        assert_eq!(iter.next(), Some((0, &0, &7.12)));
        assert_eq!(iter.next(), Some((1, &2, &4.12)));
        let mut iter_row = mat.iter_row(2);
        assert_eq!(iter_row.next(), Some((&2, &2.12)));
    }
    */

    #[test]
    fn check_sparsemat_indexlist() {
        //check_mat::<SparseMatIndexList<f32, u32>>();
        //let mut sp = SparseMatIndexList::<f32, u32>::with_capacity(3);
        let mut sp = SparseMatIndexList::<f32, u32>::with_column_track();
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
        assert_eq!(iter.next(), Some((0, &1, &4.2)));
        assert_eq!(iter.next(), Some((0, &2, &0.12)));
        assert_eq!(iter.next(), Some((0, &0, &7.12)));
        assert_eq!(iter.next(), Some((1, &2, &4.12)));
        let mut iter_row = sp.iter_row(2);
        assert_eq!(iter_row.next(), Some((&2, &2.12)));
        let sum = sp.clone() + sp.clone();
        assert_eq!(sum.get(0, 0), 14.24);
        let sub = sum.clone() - sp.clone();
        assert_eq!(sub.get(0, 0), sp.get(0, 0));
        let mul = sp.clone() * 2.0;
        assert_eq!(mul.get(0, 0), sum.get(0, 0));
        let v = vec![2.0, 4.8, 1.2];
        let mvp = sp.clone() * v;
        assert_eq!(mvp[0], 34.544);
        assert_eq!(sp.density(), 6.0 / 9.0);

        let mut iter_col = sp.iter_col(2);
        assert_eq!(iter_col.next(), Some((&1, &4.12)));
        assert_eq!(iter_col.next(), Some((&2, &2.12)));
        assert_eq!(iter_col.next(), Some((&0, &0.12)));
        assert_eq!(iter_col.next(), None);
    }

    #[test]
    fn check_sparsemat_crs() {
        let mut sp = SparseMatIndexList::<f32, u32>::with_capacity(3);
        sp.add_to(0, 1, 4.2);
        sp.add_to(2, 2, 2.12);
        sp.add_to(1, 2, 4.12);
        sp.add_to(3, 2, 1.12);
        sp.add_to(3, 3, 5.12);
        let sp_crs = SparseMatCRS::<f32, u32>::from_sparsemat_index(&sp);
        let mut iter = sp_crs.iter();
        assert_eq!(iter.next(), Some((0, &1, &4.2)));
        assert_eq!(iter.next(), Some((1, &2, &4.12)));
        assert_eq!(iter.next(), Some((2, &2, &2.12)));
        assert_eq!(iter.next(), Some((3, &2, &1.12)));
        assert_eq!(iter.next(), Some((3, &3, &5.12)));
        assert_eq!(iter.next(), None);
        let mut iter_row = sp_crs.iter_row(0);
        assert_eq!(iter_row.next(), Some((&1, &4.2)));
        assert_eq!(iter_row.next(), None);
        let mut iter_row = sp_crs.iter_row(5);
        assert_eq!(iter_row.next(), None);
        let v = vec![2.0, 4.8, 1.2, 3.4];
        let mvp = sp_crs.clone() * v;
        assert_eq!(mvp[0], 20.16);
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
        assert_eq!(iter.next(), Some((0, &1, &4.2)));
        assert_eq!(iter.next(), Some((0, &2, &0.12)));
        assert_eq!(iter.next(), Some((0, &0, &7.12)));
        assert_eq!(iter.next(), Some((1, &2, &4.12)));
        let v = vec![2.0, 4.8, 1.2];
        let mvp = sp.mvp(&v);
        assert_eq!(mvp[0], 34.544);
        assert_eq!(sp.density(), 6.0 / 9.0);
    }

    #[test]
    fn check_rowindexlist() {
        let mut list = RowIndexList::<u16>::new();
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
}
