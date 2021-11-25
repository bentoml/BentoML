from pandas.core.arrays.categorical import Categorical
from pandas.core.indexes.api import CategoricalIndex

def recode_for_groupby(
    c: Categorical, sort: bool, observed: bool
) -> tuple[Categorical, Categorical | None]:
    """
    Code the categories to ensure we can groupby for categoricals.

    If observed=True, we return a new Categorical with the observed
    categories only.

    If sort=False, return a copy of self, coded with categories as
    returned by .unique(), followed by any categories not appearing in
    the data. If sort=True, return self.

    This method is needed solely to ensure the categorical index of the
    GroupBy result has categories in the order of appearance in the data
    (GH-8868).

    Parameters
    ----------
    c : Categorical
    sort : bool
        The value of the sort parameter groupby was called with.
    observed : bool
        Account only for the observed values

    Returns
    -------
    Categorical
        If sort=False, the new categories are set to the order of
        appearance in codes (unless ordered=True, in which case the
        original order is preserved), followed by any unrepresented
        categories in the original order.
    Categorical or None
        If we are observed, return the original categorical, otherwise None
    """
    ...

def recode_from_groupby(
    c: Categorical, sort: bool, ci: CategoricalIndex
) -> CategoricalIndex:
    """
    Reverse the codes_to_groupby to account for sort / observed.

    Parameters
    ----------
    c : Categorical
    sort : bool
        The value of the sort parameter groupby was called with.
    ci : CategoricalIndex
        The codes / categories to recode

    Returns
    -------
    CategoricalIndex
    """
    ...
