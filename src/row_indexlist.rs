use crate::sparsemat::IndexType;

// Stores indices for each row for tracking the entries in the sparse matrix data vec
#[derive(Clone, Debug)]
pub(crate) struct RowIndexList<I> {
    row_start: Vec<I>,
    index_list: Vec<I>,
}

impl<I> RowIndexList<I>
where I: IndexType {
    pub(crate) const UNSET: I = <I as IndexType>::MAX;

    // Creates an empty index list
    pub(crate) fn new() -> Self {
        Self {
            row_start: Vec::<I>::new(),
            index_list: Vec::<I>::new(),
        }
    }

    // Creates an index list with reserved space
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            row_start: Vec::<I>::with_capacity(cap),
            index_list: Vec::<I>::with_capacity(cap),
        }
    }

    // Returns the number of entries in the index list
    pub(crate) fn n_entries(&self) -> usize {
        self.index_list.len()
    }

    // Returns the number of rows
    pub(crate) fn n_rows(&self) -> usize {
        self.row_start.len()
    }

    // Appends a new entry for row and returns its index in array
    pub(crate) fn push(&mut self, row: usize) -> usize {
        if row >= self.row_start.len() {
            self.row_start.resize(row + 1, Self::UNSET);
        }
        let index = I::as_indextype(self.n_entries());
        if index == Self::UNSET {
            panic!("Maximum number of {} entries reached", Self::UNSET);
        }
        self.index_list.push(Self::UNSET);
        if self.row_start[row] == Self::UNSET {
            // This is the first entry in this row
            self.row_start[row] = index;
        } else {
            // Iterate over the list of indices
            let mut iter = self.row_start[row].as_usize();
            while self.index_list[iter] != Self::UNSET {
                iter = self.index_list[iter].as_usize();
            }
            // Append the new index to the last entry in the list
            self.index_list[iter] = index;
        }
        index.as_usize()
    }

    pub(crate) fn row_iter(&self, row: usize) -> Iter<I> {
        let start = if row < self.n_rows() {
            self.row_start[row]
        } else {
            Self::UNSET
        };
        Iter::<I> {
            list: &self.index_list,
            index: start,
        }
    }
}

// Iterator returning all the index IDs of a row
pub(crate) struct Iter<'a, I> {
    list: &'a Vec<I>,
    index: I,
}

impl<'a, I> Iterator for Iter<'a, I>
where I: IndexType {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.index != RowIndexList::<I>::UNSET {
            let index_tmp = self.index.as_usize();
            self.index = self.list[index_tmp];
            Some(index_tmp)
        } else {
            None
        };
        ret
    }
}
