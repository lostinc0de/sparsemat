pub mod sparsemat_crs;
pub mod row_indexlist;
pub mod sparsemat;
pub mod sparsemat_indexlist;

#[cfg(test)]
mod tests {
    use crate::sparsemat_indexlist::*;
    use crate::sparsemat_crs::*;
    use crate::sparsemat::*;
    use crate::row_indexlist::*;

    #[test]
    fn check_sparsemat_indexlist() {
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
        let mut iter_col = sp.row_iter_columns(0);
        let mut iter_val = sp.row_iter_values(0);
        assert_eq!(iter_col.next(), Some(1));
        assert_eq!(iter_val.next(), Some(&4.2));
        assert_eq!(iter_col.next(), Some(2));
        assert_eq!(iter_val.next(), Some(&0.12));
        assert_eq!(iter_col.next(), Some(0));
        assert_eq!(iter_val.next(), Some(&7.12));
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
    }

    #[test]
    fn check_sparsemat_crs() {
        let mut sp = SparseMatIndexList::<f32, u32>::with_capacity(3);
        sp.add_to(0, 1, 4.2);
        sp.add_to(1, 2, 4.12);
        sp.add_to(2, 2, 2.12);
        sp.add_to(3, 2, 1.12);
        sp.add_to(3, 3, 5.12);
        let sp_crs = SparseMatCRS::<f32, u32>::from_sparsemat_index(&sp);
        let mut iter_col = sp_crs.row_iter_columns(0);
        let mut iter_val = sp_crs.row_iter_values(0);
        assert_eq!(iter_col.next(), Some(1));
        assert_eq!(iter_val.next(), Some(&4.2));
        assert_eq!(iter_col.next(), None);
        assert_eq!(iter_val.next(), None);
        let mut iter_col = sp_crs.row_iter_columns(3);
        let mut iter_val = sp_crs.row_iter_values(3);
        assert_eq!(iter_col.next(), Some(2));
        assert_eq!(iter_val.next(), Some(&1.12));
        assert_eq!(iter_col.next(), Some(3));
        assert_eq!(iter_val.next(), Some(&5.12));
        let v = vec![2.0, 4.8, 1.2, 3.4];
        let mvp = sp_crs.clone() * v;
        assert_eq!(mvp[0], 20.16);
        assert_eq!(sp_crs.density(), 5.0 / 16.0);
    }

    #[test]
    fn check_rowindexlist() {
        let mut list = RowIndexList::<u16>::new();
        list.push(1);
        list.push(1);
        list.push(2);
        list.push(4);
        list.push(1);
        assert_eq!(list.row_iter(0).next(), None);
        assert_eq!(list.n_entries(), 5);
        let mut iter = list.row_iter(1);
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(4));
    }
}
