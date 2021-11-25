

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from packaging.version import Version
from transformers.pipelines import Pipeline
from transformers.tokenization_utils import BatchEncoding

ORT_QUANTIZE_MINIMUM_VERSION = ...
SUPPORTED_PIPELINES = ...
class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """
    def __init__(self) -> None:
        ...
    


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath

    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns: String with concatenated identifier at the end of the filename
    """
    ...

def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    ...

def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any Non

    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    ...

def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    """
    Attempt to infer the static vs dynamic axes for each input and output tensors for a specific model

    Args:
        nlp: The pipeline object holding the model to be exported
        framework: The framework identifier to dispatch to the correct inference scheme (pt/tf)

    Returns:

        - List of the inferred input variable names
        - List of the inferred output variable names
        - Dictionary with input/output variables names as key and shape tensor as value
        - a BatchEncoding reference which was used to infer all the above information
    """
    ...

def load_graph_from_args(pipeline_name: str, framework: str, model: str, tokenizer: Optional[str] = ..., **models_kwargs) -> Pipeline:
    """
    Convert the set of arguments provided through the CLI to an actual pipeline reference (tokenizer + model

    Args:
        pipeline_name: The kind of pipeline to use (ner, question-answering, etc.)
        framework: The actual model to convert the pipeline from ("pt" or "tf")
        model: The model name which will be loaded by the pipeline
        tokenizer: The tokenizer name which will be loaded by the pipeline, default to the model's value

    Returns: Pipeline object

    """
    ...

def convert_pytorch(nlp: Pipeline, opset: int, output: Path, use_external_format: bool):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB

    Returns:

    """
    ...

def convert_tensorflow(nlp: Pipeline, opset: int, output: Path):
    """
    Export a TensorFlow backed pipeline to ONNX Intermediate Representation (IR

    Args:
        nlp: The pipeline to be exported
        opset: The actual version of the ONNX operator set to use
        output: Path where will be stored the generated ONNX model

    Notes: TensorFlow cannot export model bigger than 2GB due to internal constraint from TensorFlow

    """
    ...

def convert(framework: str, model: str, output: Path, opset: int, tokenizer: Optional[str] = ..., use_external_format: bool = ..., pipeline_name: str = ..., **model_kwargs):
    """
    Convert the pipeline object to the ONNX Intermediate Representation (IR) format

    Args:
        framework: The framework the pipeline is backed by ("pt" or "tf")
        model: The name of the model to load for the pipeline
        output: The path where the ONNX graph will be stored
        opset: The actual version of the ONNX operator set to use
        tokenizer: The name of the model to load for the pipeline, default to the model's name if not provided
        use_external_format: Split the model definition from its parameters to allow model bigger than 2GB (PyTorch only)
        pipeline_name: The kind of pipeline to instantiate (ner, question-answering, etc.)
        model_kwargs: Keyword arguments to be forwarded to the model constructor

    Returns:

    """
    ...

def optimize(onnx_model_path: Path) -> Path:
    """
    Load the model at the specified path and let onnxruntime look at transformations on the graph to enable all the
    optimizations possibl

    Args:
        onnx_model_path: filepath where the model binary description is stored

    Returns: Path where the optimized model binary description has been saved

    """
    ...

def quantize(onnx_model_path: Path) -> Path:
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU

    Args:
        onnx_model_path: Path to location the exported ONNX model is stored

    Returns: The Path generated for the quantized
    """
    ...

def verify(path: Path):
    ...

if __name__ == "__main__":
    parser = ...
    args = ...
