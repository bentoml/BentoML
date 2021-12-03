from .base_operator import QuantOperatorBase

class GatherQuant(QuantOperatorBase):
    def __init__(self, onnx_quantizer, onnx_node) -> None: ...
