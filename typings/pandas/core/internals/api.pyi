from pandas._libs.internals import BlockPlacement
from pandas._typing import Dtype
from pandas.core.internals.blocks import Block

"""
This is a pseudo-public API for downstream libraries.  We ask that downstream
authors

1) Try to avoid using internals directly altogether, and failing that,
2) Use only functions exposed here (or in core.internals)

"""

def make_block(
    values, placement, klass=..., ndim=..., dtype: Dtype | None = ...
) -> Block:
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
    ...

def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placment and values.
    """
    ...
