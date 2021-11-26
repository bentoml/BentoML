from typing import TYPE_CHECKING
from pandas.core.internals.managers import BlockManager

if TYPE_CHECKING: ...
BlockPairInfo = ...

def operate_blockwise(
    left: BlockManager, right: BlockManager, array_op
) -> BlockManager: ...
def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool: ...
