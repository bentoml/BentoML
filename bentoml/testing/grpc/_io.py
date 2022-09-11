from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from bentoml.exceptions import BentoMLException
from bentoml._internal.utils import LazyLoader

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from bentoml.grpc.v1alpha1 import service_pb2 as pb
else:
    from bentoml.grpc.utils import import_generated_stubs

    pb, _ = import_generated_stubs()
    np = LazyLoader("np", globals(), "numpy")


def randomize_pb_ndarray(shape: tuple[int, ...]) -> pb.NDArray:
    arr: NDArray[np.float32] = t.cast("NDArray[np.float32]", np.random.rand(*shape))
    return pb.NDArray(
        shape=list(shape), dtype=pb.NDArray.DTYPE_FLOAT, float_values=arr.ravel()
    )


def make_pb_ndarray(arr: NDArray[t.Any]) -> pb.NDArray:
    from bentoml._internal.io_descriptors.numpy import npdtype_to_dtypepb_map
    from bentoml._internal.io_descriptors.numpy import npdtype_to_fieldpb_map

    try:
        fieldpb = npdtype_to_fieldpb_map()[arr.dtype]
        dtypepb = npdtype_to_dtypepb_map()[arr.dtype]
        return pb.NDArray(
            **{
                fieldpb: arr.ravel().tolist(),
                "dtype": dtypepb,
                "shape": tuple(arr.shape),
            },
        )
    except KeyError:
        raise BentoMLException(
            f"Unsupported dtype '{arr.dtype}' for response message.",
        ) from None
