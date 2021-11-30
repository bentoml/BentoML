from typing import List
from numpy.core.fromnumeric import (
    amax,
    amin,
    argmax,
    argmin,
    cumprod,
    cumsum,
    mean,
    prod,
    std,
    sum,
    var,
)
from numpy.lib.function_base import median, percentile, quantile

__all__: List[str]
nanmin = amin
nanmax = amax
nanargmin = argmin
nanargmax = argmax
nansum = sum
nanprod = prod
nancumsum = cumsum
nancumprod = cumprod
nanmean = mean
nanvar = var
nanstd = std
nanmedian = median
nanpercentile = percentile
nanquantile = quantile
