from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.api import CategoricalIndex

def recode_for_groupby(
    c: Categorical, sort: bool, observed: bool
) -> tuple[Categorical, Categorical | None]: ...
def recode_from_groupby(
    c: Categorical, sort: bool, ci: CategoricalIndex
) -> CategoricalIndex: ...
