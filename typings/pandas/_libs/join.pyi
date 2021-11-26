import numpy as np

def inner_join(
    left: np.ndarray, right: np.ndarray, max_groups: int
) -> tuple[np.ndarray, np.ndarray]: ...
def left_outer_join(
    left: np.ndarray, right: np.ndarray, max_groups: int, sort: bool = ...
) -> tuple[np.ndarray, np.ndarray]: ...
def full_outer_join(
    left: np.ndarray, right: np.ndarray, max_groups: int
) -> tuple[np.ndarray, np.ndarray]: ...
def ffill_indexer(indexer: np.ndarray) -> np.ndarray: ...
def left_join_indexer_unique(left: np.ndarray, right: np.ndarray) -> np.ndarray: ...
def left_join_indexer(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def inner_join_indexer(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def outer_join_indexer(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def asof_join_backward_on_X_by_Y(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_by_values: np.ndarray,
    right_by_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
def asof_join_forward_on_X_by_Y(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_by_values: np.ndarray,
    right_by_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
def asof_join_nearest_on_X_by_Y(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_by_values: np.ndarray,
    right_by_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
def asof_join_backward(
    left_values: np.ndarray,
    right_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
def asof_join_forward(
    left_values: np.ndarray,
    right_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
def asof_join_nearest(
    left_values: np.ndarray,
    right_values: np.ndarray,
    allow_exact_matches: bool = ...,
    tolerance=...,
) -> tuple[np.ndarray, np.ndarray]: ...
