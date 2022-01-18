import sys

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


# Mock external libraries to avoid doc dependencies
import mock
import bentoml._internal.utils.pkg as pkg

for mod_name in autodoc_mock_imports:
    sys.modules[mod_name] = mock.MagicMock()
    sys.modules[mod_name].__version__ = mock.Mock()
    sys.modules[mod_name].__spec__ = mock.Mock()

sys.modules['easyocr'].__version__ = "1.4"
pkg.get_pkg_version = mock.MagicMock()
    

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


def setup(app):
    app.add_css_file("css/bentoml.css")
