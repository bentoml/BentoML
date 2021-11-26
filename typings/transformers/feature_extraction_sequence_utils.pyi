from typing import Dict, List, Optional, Union
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .file_utils import PaddingStrategy, TensorType

logger = ...

class SequenceFeatureExtractor(FeatureExtractionMixin):
    def __init__(
        self, feature_size: int, sampling_rate: int, padding_value: float, **kwargs
    ) -> None: ...
    def pad(
        self,
        processed_features: Union[
            BatchFeature,
            List[BatchFeature],
            Dict[str, BatchFeature],
            Dict[str, List[BatchFeature]],
            List[Dict[str, BatchFeature]],
        ],
        padding: Union[bool, str, PaddingStrategy] = ...,
        max_length: Optional[int] = ...,
        pad_to_multiple_of: Optional[int] = ...,
        return_attention_mask: Optional[bool] = ...,
        return_tensors: Optional[Union[str, TensorType]] = ...,
    ) -> BatchFeature: ...
