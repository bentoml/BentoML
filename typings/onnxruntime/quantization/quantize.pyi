from pathlib import Path
from .calibrate import CalibrationDataReader

def optimize_model(model_path: Path): ...
def load_model(model_path: Path, optimize=...): ...
def quantize(
    model,
    per_channel=...,
    nbits=...,
    quantization_mode=...,
    static=...,
    force_fusions=...,
    symmetric_activation=...,
    symmetric_weight=...,
    quantization_params=...,
    nodes_to_quantize=...,
    nodes_to_exclude=...,
    op_types_to_quantize=...,
): ...
def quantize_static(
    model_input,
    model_output,
    calibration_data_reader: CalibrationDataReader,
    quant_format=...,
    op_types_to_quantize=...,
    per_channel=...,
    reduce_range=...,
    activation_type=...,
    weight_type=...,
    nodes_to_quantize=...,
    nodes_to_exclude=...,
    optimize_model=...,
    use_external_data_format=...,
    calibrate_method=...,
    extra_options=...,
): ...
def quantize_dynamic(
    model_input: Path,
    model_output: Path,
    op_types_to_quantize=...,
    per_channel=...,
    reduce_range=...,
    activation_type=...,
    weight_type=...,
    nodes_to_quantize=...,
    nodes_to_exclude=...,
    optimize_model=...,
    use_external_data_format=...,
    extra_options=...,
): ...
def quantize_qat(
    model_input: Path,
    model_output: Path,
    op_types_to_quantize=...,
    per_channel=...,
    reduce_range=...,
    activation_type=...,
    weight_type=...,
    nodes_to_quantize=...,
    nodes_to_exclude=...,
    use_external_data_format=...,
): ...
