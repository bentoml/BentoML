from datetime import datetime

# Adding BentoML source directory for accessing BentoML version
import bentoml

# -- Project information -----------------------------------------------------

project = "BentoML"
copyright = f"2022-{datetime.now().year}, bentoml.com"
author = "bentoml.com"
version = bentoml.__version__

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# See https://github.com/readthedocs/readthedocs.org/issues/2149
master_doc = "index"

# Sphinx extensions
extensions = [
    "sphinxext.opengraph",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.ifconfig",
    "sphinx_click.ext",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "sphinxcontrib.spelling",
    "myst_parser",
    "sphinx_inline_tabs",
]

# Plugin Configurations:
napoleon_include_private_with_doc = False
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = False

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 10

ogp_site_url = "http://docs.bentoml.org"
ogp_image = "https://docs.bentoml.org/en/latest/_images/bentoml-readme-header.jpeg"
ogp_site_name = "BentoML Documentation"
ogp_use_first_image = True

issues_default_group_project = "bentoml/bentoml"

todo_include_todos = True

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "zenburn"
pygments_dark_style = "monokai"

myst_enable_extensions = ["colon_fence"]

# Remove the prompt when copying examples
copybutton_prompt_text = r">>> |\.\.\.|» |$ |% "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#44a4c6 ",
        "color-brand-content": "#44a4c6 ",
    },
    "dark_css_variables": {
        "color-brand-primary": "#c9378a ",
        "color-brand-content": "#c9378a ",
    },
    "source_repository": "https://github.com/bentoml/bentoml/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bentoml/bentoml",
            "html": " 🍱 ",
            "class": "",
        },
    ],
}

html_title = "BentoML"
html_logo = "_static/img/logo.svg"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_js_files = ["js/custom.js"]
html_show_sphinx = False
html_favicon = "_static/img/favicon-32x32.ico"

# Private dictionary for spell checker
spelling_word_list_filename = ["bentoml_wordlist.txt"]

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
