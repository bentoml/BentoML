from datetime import datetime

from pygments.lexers import PythonLexer

# Adding BentoML source directory for accessing BentoML version
import bentoml

# -- Project information -----------------------------------------------------

project = "BentoML"
copyright = f"2020-{datetime.now().year}, bentoml.org"
author = "bentoml.org"
version = bentoml.__version__

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# See https://github.com/readthedocs/readthedocs.org/issues/2149
master_doc = "index"

# Sphinx extensions
extensions = [
    "sphinx_tabs.tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_click.ext",
    "sphinx_copybutton",
    "recommonmark",
    "sphinxcontrib.spelling",
]

napoleon_include_private_with_doc = False
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = False

todo_include_todos = True

sphinx_tabs_disable_css_loading = True

# mock any heavy imports, eg: imports from frameworks library
autodoc_mock_imports = [
    "catboost",
    "coremltools",
    "detectron2",
    "detectron2.config",
    "detectron2.modeling",
    "detectron2.checkpoint",
    "torch",
    "torchvision",
    "torchtext",
    "easyocr",
    "evalml",
    "fastai",
    "fasttext",
    "mxnet",
    "mxnet.gluon",
    "h2o",
    "h2o.model",
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.python.client",
    "tensorflow.python.training.tracking.tracking",
    "tensorflow_hub",
    "keras",
    "trax",
    "flax",
    "jax",
    "lightgbm",
    "mlflow",
    "mlflow.pyfunc",
    "mlflow.tracking.artifact_utils",
    "onnx",
    "onnxruntime",
    "PyRuntime",
    "paddle",
    "paddle.nn",
    "paddle.inference",
    "paddle.fluid",
    "paddlehub",
    "paddlehub.module.manager",
    "paddlehub.server.server",
    "pycaret",
    "pycaret.internal.tabular",
    "pyspark",
    "pytorch",
    "torch.nn.parallel",
    "pytorch_lightning",
    "sklearn",
    "joblib",
    "spacy",
    "spacy.util",
    "thinc.util",
    "thinc.backends",
    "statsmodels",
    "statsmodels.api",
    "statsmodels.tools.parallel",
    "transformers",
    "transformers.file_utils",
    "xgboost",
]

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "zenburn"

# Remove the prompt when copying examples
copybutton_prompt_text = r">>> |\.\.\.|» |$ |% "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "logo_only": False,
    "display_version": True,
    "navigation_depth": -1,
}

# Static folder path
html_static_path = ["_static"]

# Private dictionary for spell checker
spelling_word_list_filename = ["bentoml_wordlist.txt"]


# -- Custom lexers
class TensorflowV1Lexer(PythonLexer):
    name = "Tensorflow V1"
    aliases = ["tensorflow_v1"]


class TensorflowV2Lexer(PythonLexer):
    name = "Tensorflow V2"
    aliases = ["tensorflow_v2"]


class KerasTensorflowV1Lexer(PythonLexer):
    name = "Keras (Tensorflow V1 backend)"
    aliases = ["keras_v1"]


class KerasTensorflowV2Lexer(PythonLexer):
    name = "Keras (Tensorflow V2 backend)"
    aliases = ["keras_v2"]


def setup(app):
    # css files
    app.add_css_file("css/typeface/typeface.css")
    app.add_css_file("css/common.css")
    app.add_css_file("css/tabs.css")
    app.add_css_file("css/nav_patch.css")
    app.add_css_file("css/bentoml.css")

    # js files
    app.add_js_file("js/bentoml.js")

    # Adding lexers for rendering different code version
    app.add_lexer("keras_v1", KerasTensorflowV1Lexer)
    app.add_lexer("keras_v2", KerasTensorflowV2Lexer)
    app.add_lexer("tensorflow_v1", TensorflowV1Lexer)
    app.add_lexer("tensorflow_v2", TensorflowV2Lexer)
