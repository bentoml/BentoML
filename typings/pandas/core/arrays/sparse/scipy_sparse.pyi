"""
Interaction with scipy.sparse matrices.

Currently only includes to_coo helpers.
"""

def sparse_series_to_coo(
    ss, row_levels=..., column_levels=..., sort_labels=...
):  # -> tuple[Unknown, list[Any] | Any, list[Any] | Any]:
    """
    Convert a sparse Series to a scipy.sparse.coo_matrix using index
    levels row_levels, column_levels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels.
    """
    ...

def coo_to_sparse_series(A, dense_index: bool = ...):  # -> Series | Any | DataFrame:
    """
    Convert a scipy.sparse.coo_matrix to a SparseSeries.

    Parameters
    ----------
    A : scipy.sparse.coo.coo_matrix
    dense_index : bool, default False

    Returns
    -------
    Series

    Raises
    ------
    TypeError if A is not a coo_matrix
    """
    ...
