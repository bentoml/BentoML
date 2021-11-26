from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase

class Direct8BitOp(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
    def quantize(self): ...

class QDQDirect8BitOp(QDQOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
