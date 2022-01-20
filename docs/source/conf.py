from pygments.lexers import PythonLexer

# Adding BentoML source directory for accessing BentoML version
import bentoml

# -- Project information -----------------------------------------------------

project = "BentoML"
copyright = "2021, bentoml.org"
author = "bentoml.org"
version = bentoml.__version__

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# See https://github.com/readthedocs/readthedocs.org/issues/2149
master_doc = "index"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_click.ext",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "recommonmark",
    "sphinxcontrib.spelling",
]

napoleon_include_private_with_doc = False
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = False

todo_include_todos = True

# mock any heavy imports, eg: imports from frameworks library
autodoc_mock_imports = [
    "catboost",
    "coremltools",
    "detectron2",
    "torch",
    "torchvision",
    "torchtext",
    "easyocr",
    "evalml",
    "fastai",
    "fasttext",
    "mxnet",
    "h2o",
    "tensorflow",
    "tensorflow_hub",
    "keras",
    "trax",
    "flax",
    "jax",
    "lightgbm",
    "mlflow",
    "onnx",
    "onnxruntime",
    "PyRuntime",
    "paddle",
    "paddlehub",
    "pycaret",
    "pyspark",
    "pytorch",
    "pytorch_lightning",
    "sklearn",
    "joblib",
    "spacy",
    "thinc",
    "statsmodels",
    "transformers",
    "xgboost",
]

autodoc_typehints = "signature"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Remove the prompt when copying examples
copybutton_prompt_text = r">>> |\.\.\.|» |$ "
copybutton_prompt_is_regexp = True

# Exclude directory pattern
exclude_patterns = ["guides/configs/*.md"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    # "collapse_navigation": True,
    "logo_only": True,
    "display_version": True,
}

html_logo = "_static/img/bentoml-removebg.png"

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


def setup(app):
    app.add_css_file("css/bentoml.css")
    # Adding lexers for rendering different code version
    app.add_lexer("tensorflow_v1", TensorflowV1Lexer)
    app.add_lexer("tensorflow_v2", TensorflowV2Lexer)
