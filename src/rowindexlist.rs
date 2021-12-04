use crate::types::IndexType;

// Stores positions of each row for tracking the entries in the sparse matrix data vec
// The vec index_list contains the next positions in the list while the vec pos_start
// contains the starting position for each row
// The position in the index_list vec is the actual index
// Example: row index:    0  1    2    3  4
//          pos_start  = [0, 2, UNSET, 4, 3]
//          index_list = [1, UNSET, UNSET, 5, 6, UNSET, UNSET]
//          -> First row holds two entries at position 0 and 1
//          -> Second row only holds one entry at position 2
//          -> Third row is empty
//          -> Fourth row holds two entries at position 4 and 6
//          -> Fifth row holds two entries at position 3 and 5
//             and has been inserted before the fourth row
// Iterating over the entries of the fourth row would look like this:
// pos_start =  [0, 2, UNSET, 4, ...]
//                            |
//                            -->--
//                                |
// index_list = [1, UNSET, UNSET, 5, 6, UNSET, UNSET]
//                                |       |
//                                ---->----
// So the entries for the fourth row are stored at position 4 and 5 in the data vec
#[derive(Clone, Debug)]
pub(crate) struct IndexList<I> {
    pos_start: Vec<I>,
    index_list: Vec<I>,
}

impl<I> IndexList<I>
where I: IndexType {
    pub(crate) const UNSET: I = <I as IndexType>::MAX;

    // Creates an empty index list
    pub(crate) fn new() -> Self {
        Self {
            pos_start: Vec::<I>::new(),
            index_list: Vec::<I>::new(),
        }
    }

    // Creates an index list with reserved space
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            pos_start: Vec::<I>::with_capacity(cap),
            index_list: Vec::<I>::with_capacity(cap),
        }
    }

    // Returns the number of entries in the index list
    pub(crate) fn n_entries(&self) -> usize {
        self.index_list.len()
    }

    // Returns the number of rows
    pub(crate) fn n_rows(&self) -> usize {
        self.pos_start.len()
    }

    // Appends a new entry for row and returns its index in array
    pub(crate) fn push(&mut self, row: usize) -> usize {
        if row >= self.pos_start.len() {
            self.pos_start.resize(row + 1, Self::UNSET);
        }
        let index = I::as_indextype(self.n_entries());
        // Check if the maximum number of entries has been reached - This should never happen
        assert_ne!(index, Self::UNSET);
        self.index_list.push(Self::UNSET);
        if self.pos_start[row] == Self::UNSET {
            // This is the first entry in this row
            self.pos_start[row] = index;
        } else {
            // Iterate over the list of indices
            let mut iter = self.pos_start[row].as_usize();
            while self.index_list[iter] != Self::UNSET {
                iter = self.index_list[iter].as_usize();
            }
            // Append the new index to the last entry in the list
            self.index_list[iter] = index;
        }
        index.as_usize()
    }

    pub(crate) fn iter_row(&self, row: usize) -> IterRow<I> {
        IterRow::<I> {
            list: self,
            pos: self.pos_start[row],
        }
    }
}

// Iterator returning all the index IDs of a row
pub(crate) struct IterRow<'a, I> {
    list: &'a IndexList<I>,
    pos: I,
}

impl<'a, I> Iterator for IterRow<'a, I>
where I: IndexType {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos != IndexList::<I>::UNSET {
            let index = self.pos.as_usize();
            self.pos = self.list.index_list[index];
            Some(index)
        } else {
            None
        }
    }
}
