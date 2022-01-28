from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typing as t
    from typing import Literal

    from numpy import generic as NpGeneric
    from pandas import Series as PdSeries  # type: ignore[reportMissingTypeStubs]
    from pandas import DataFrame as PdDataFrame  # type: ignore[reportMissingTypeStubs]
    from numpy.typing import NDArray as NpNDArray
    from pyarrow.plasma import ObjectID
    from pyarrow.plasma import PlasmaClient
    from transformers.modeling_utils import PreTrainedModel
    from transformers.pipelines.base import Pipeline as TransformersPipeline
    from transformers.modeling_tf_utils import TFPreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.modeling_flax_utils import FlaxPreTrainedModel
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    from transformers.feature_extraction_sequence_utils import (
        SequenceFeatureExtractor as PreTrainedFeatureExtractor,
    )

    TransformersModelType = t.Union[
        PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel
    ]
    TransformersTokenizerType = t.Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

    DataFrameOrient = Literal["split", "records", "index", "columns", "values", "table"]
    SeriesOrient = Literal["split", "records", "index", "table"]

    from starlette.types import ASGIApp

    class AsgiMiddleware(t.Protocol):
        def __call__(self, app: ASGIApp, **options: t.Any) -> ASGIApp:
            ...

    __all__ = [
        "PdSeries",
        "PdDataFrame",
        "NpNDArray",
        "ObjectID",
        "PlasmaClient",
        "NpGeneric",
        "DataFrameOrient",
        "SeriesOrient",
        "AsgiMiddleware",
        "ASGIApp",
        # transformers-related types
        "TransformersPipeline",
        "TransformersModelType",
        "TransformersTokenizerType",
        "PreTrainedFeatureExtractor",
    ]
