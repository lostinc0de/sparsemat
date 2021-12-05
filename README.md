# sparsemat: A sparse matrix library written in Rust

## Usage
The goal of this project is to provide an efficient and easy to use sparse matrix library in Rust.
Different implementations share the same interface SparseMatrix and have to be row-wise e.g. a row iterator is provided.
If a column iterator is necessary, the IterColumn trait may be used if available. 
The common sparse matrix format CRS (Compressed Row Storage) is used.
However, this format should not be used to assemble a sparse matrix, since inserting entries may be expensive and take O(N) time in the worst case.
For this purpose another implementation is provided: SparseMatIndexList.
It uses an index-list to track all the entries which are just appended to a data vector.
This format needs more space and may be slower than CRS, but entries are inserted in O(1) time.
To take advantage of both formats the SparseMatIndexList should be used to assemble the matrix and can be converted to SparseMatCRS afterwards:

```rust
    use sparsemat::SparseMatIndexList;
    use sparsemat::SparseMatCRS;

    let mut sp = SparseMatIndexList::<f32, u32>::with_capacity(3);
    // Add entries and modify them
    sp.add_to(0, 1, 4.2);
    sp.set(1, 2, 4.12);
    *sp.get_mut(0, 2) += 0.12;
    // Convert to CRS format
    let sp_crs = sp.to_crs();
    // Use matrix here
```
