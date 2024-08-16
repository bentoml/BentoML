import importlib.metadata
from datetime import datetime

# -- Project information -----------------------------------------------------

project = "BentoML"
copyright = f"2022-{datetime.now().year}, bentoml.com"
author = "bentoml.com"

version = importlib.metadata.version("bentoml")

# -- General configuration ---------------------------------------------------

source_suffix = [".rst", ".md"]

# See https://github.com/readthedocs/readthedocs.org/issues/2149
master_doc = "index"

# exclude patterns
exclude_patterns = [
    "**/*/bazel-*",  # generated by bazel
    "**/*/node_modules/*",  # node_modules
    "**/*/.build/*",  # generated by swift
    "**/*/thirdparty/*",  # generated by swift
]

# Sphinx extensions
extensions = [
    "sphinxext.opengraph",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx_click",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "sphinxcontrib.spelling",
    "myst_parser",
    "sphinx_inline_tabs",
    "hoverxref.extension",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pip": ("https://pip.pypa.io/en/latest", None),
}
extlinks = {
    "pypi": ("https://pypi.org/project/%s", "%s"),  # noqa: WPS323
    "wiki": ("https://wikipedia.org/wiki/%s", "%s"),  # noqa: WPS323
    "github": ("https://github.com/%s", "%s"),  # noqa: WPS323
    "examples": (
        "https://github.com/bentoml/BentoML/tree/main/examples/%s",
        "examples/",
    ),  # noqa: WPS323
}
# custom roles
rst_prolog = """
.. role:: raw-html(raw)
    :format: html
"""
# hoverxref settings
hoverxref_auto_ref = True
hoverxref_sphinxtabs = True
hoverxref_role_types = {
    "hoverxref": "modal",
    "ref": "tooltip",
    "mod": "tooltip",
    "class": "tooltip",
    "doc": "tooltip",
}
hoverxref_intersphinx = ["python", "pip"]

# Plugin Configurations:
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_special_with_doc = False
napoleon_attr_annotations = True
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_typehints_description_target = "documented"

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 10

ogp_site_url = "http://docs.bentoml.com"
ogp_image = "https://docs.bentoml.com/en/latest/_static/img/bentoml-banner.jpg"
ogp_site_name = "BentoML Documentation"
ogp_use_first_image = True

issues_default_group_project = "bentoml/bentoml"

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "zenburn"
pygments_dark_style = "monokai"

myst_enable_extensions = ["colon_fence"]

# Remove the prompt when copying examples
copybutton_prompt_text = r">>> |\.\.\.|> |» |\% |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_theme_options = {
    "announcement": "This is the latest BentoML documentation. For BentoML 1.1.11 and older versions, see the <a href='https://docs.bentoml.com/en/v1.1.11/'>previous documentation</a>.",
    "light_css_variables": {
        "color-brand-primary": "#4dad8c ",
        "color-brand-content": "#4dad8c ",
        "color-announcement-background": "#4dad8c ",
        "color-announcement-text": "#ffffff ",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4dad8c ",
        "color-brand-content": "#4dad8c ",
        "color-announcement-background": "#4dad8c ",
        "color-announcement-text": "#ffffff ",
    },
    "source_repository": "https://github.com/bentoml/bentoml/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/bentoml",
            "html": "&nbsp;&nbsp;",
            "class": "fab fa-github",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/company/bentoml/",
            "html": "&nbsp;&nbsp;",
            "class": "fab fa-linkedin",
        },
        {
            "name": "X",
            "url": "https://twitter.com/bentomlai",
            "html": "&nbsp;&nbsp;",
            "class": "fab fa-x-twitter",
        },
        {
            "name": "Slack",
            "url": "https://l.bentoml.com/join-slack",
            "html": "&nbsp;&nbsp;",
            "class": "fab fa-slack",
        },
    ],
    "light_logo": "img/logo-light.svg",
    "dark_logo": "img/logo-dark.svg",
}

html_title = "BentoML"
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.6.0/css/all.min.css",
]
html_js_files = ["js/custom.js"]
html_show_sphinx = False
html_favicon = "_static/img/favicon-32x32.png"

# Private dictionary for spell checker
spelling_word_list_filename = ["bentoml_wordlist.txt"]

# mock any heavy imports, eg: imports from frameworks library
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "diffusers",
    "detectron2",
    "easyocr",
    "flax",
    "jax",
    "jaxlib",
    "torchtext",
    "fastai",
    "fastai.learner.Learner",
    "tensorflow",
    "tensorflow.keras",
    "tensorflow.python.client",
    "tensorflow.python.training.tracking.tracking",
    "keras",
    "lightgbm",
    "mlflow",
    "onnx",
    "onnxruntime",
    "torch.nn.parallel",
    "pytorch_lightning",
    "sklearn",
    "joblib",
    "transformers",
    "transformers.file_utils",
    "xgboost",
    "catboost",
    "prometheus_client",
    "bentoml._internal.models.model.ModelSignatureDict",
]
