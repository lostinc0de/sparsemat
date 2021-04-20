use crate::sparsemat::IndexType;

// Stores indices for each row for tracking the entries in the sparse matrix data vec
// The vec index_list contains the next positions in the list while the vec row_start
// contains the starting position for each row
// The position in the index_list vec is the actual index
// Example: row_start  = [0, 2, UNSET, 4, 3]
//          index_list = [1, UNSET, UNSET, 5, 6, UNSET, UNSET]
//          -> First row holds two entries at position 0 and 1
//          -> Second row only holds one entry at position 2
//          -> Third row is empty
//          -> Fourth row holds two entries at position 4 and 6
//          -> Fifth row holds two entries at position 3 and 5
//             and has been inserted before the fourth row
// Iterating over the entries of the fourth row would be look like this:
// row_start =  [0, 2, UNSET, 4, ...]
//                            |
//                            -->--
//                                |
// index_list = [1, UNSET, UNSET, 5, 6, UNSET, UNSET]
//                                |       |
//                                ---->----
// So the entries for the fourth row are stored at position 4 and 5 in the data vec
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

    pub(crate) fn iter_row(&self, row: usize) -> IterRow<I> {
        let start = if row < self.n_rows() {
            self.row_start[row]
        } else {
            Self::UNSET
        };
        IterRow::<I> {
            list: self,
            pos: start,
        }
    }

    pub(crate) fn iter(&self) -> Iter<I> {
        Iter::<I> {
            list: self,
            row: 0,
            pos: self.row_start[0],
        }
    }
}

// Iterator returning all the index IDs of a row
pub(crate) struct IterRow<'a, I> {
    list: &'a RowIndexList<I>,
    pos: I,
}

impl<'a, I> Iterator for IterRow<'a, I>
where I: IndexType {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos != RowIndexList::<I>::UNSET {
            let index = self.pos.as_usize();
            self.pos = self.list.index_list[index];
            Some(index)
        } else {
            None
        }
    }
}

// Iterator returning the row and all the index IDs
pub(crate) struct Iter<'a, I> {
    list: &'a RowIndexList<I>,
    row: usize,
    pos: I,
}

impl<'a, I> Iterator for Iter<'a, I>
where I: IndexType {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Increment row if necessary
        while self.pos == RowIndexList::<I>::UNSET
               && self.row < (self.list.n_rows() - 1) {
            self.row += 1;
            self.pos = self.list.row_start[self.row];
        }
        if self.pos != RowIndexList::<I>::UNSET {
            let index = self.pos.as_usize();
            self.pos = self.list.index_list[index];
            Some((self.row, index))
        } else {
            None
        }
    }
}
