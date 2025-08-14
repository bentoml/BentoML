# ruff: noqa
import os
import sys
import typing as t
import logging
import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
from pathlib import Path

import yaml
import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException, MissingDependencyException
import spacy
from simple_di import inject, Provide
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models import ModelStore


import importlib.metadata as importlib_metadata

_spacy_version = importlib_metadata.version("spacy")

_check_compat = _spacy_version.startswith("3")
if not _check_compat:
    raise EnvironmentError(
        "BentoML only provides support for SpaCy 3.x and above"
        " as this allows more support for SpaCy's design. Currently"
        f" detected spacy to have version {_spacy_version}"
    )

from bentoml.exceptions import InvalidArgument
from bentoml._internal.utils.lazy_loader import LazyLoader

util = LazyLoader("util", globals(), "spacy.util")
thinc_util = LazyLoader("thinc_util", globals(), "thinc.util")
thinc_backends = LazyLoader("thinc_backends", globals(), "thinc.backends")

torch = LazyLoader("torch", globals(), "torch")
tensorflow = LazyLoader("tensorflow", globals(), "tensorflow")

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.vocab import Vocab
    from thinc.config import Config
    from spacy.tokens.doc import Doc

logger = logging.getLogger(__name__)

# Define projects commands that are not supported
PROJECTS_CMD_NOT_SUPPORTED = [
    "assets",
    "document",
    "dvc",
    "push",
    "run",
]


def save(
    name: str,
    model: "Language",
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> bentoml.Model:
    from bentoml._internal.models.model import ModelContext

    context = ModelContext(
        framework_name="spacy",
        framework_versions={"spacy": _spacy_version},
    )

    signatures = {
        "__call__": {"batchable": True},
        "pipe": {"batchable": True},
    }

    meta = model.meta
    pip_package = f"{meta['lang']}_{meta['name']}"

    options = {
        "pip_package": pip_package,
    }
    if "requirements" in meta:
        options["additional_requirements"] = meta["requirements"]

    with bentoml.models.create(
        name,
        module=__name__,
        signatures=signatures,
        options=options,
        context=context,
        metadata=metadata,
    ) as bento_model:
        model_path = bento_model.path
        model.to_disk(model_path)

    return bento_model


def load_project(
    bento_model: bentoml.Model,
) -> str:
    if "projects_uri" in bento_model.info.options:
        logger.warning(
            "We will only return the path of projects saved under BentoML model store"
            " and leave projects interaction for users to determine. "
            "Refer to https://spacy.io/api/cli#project for more information."
        )
        return os.path.join(bento_model.path, bento_model.info.options["target_path"])

    raise EnvironmentError(
        "Cannot use `bentoml.spacy.load_project()` to load non-SpaCy Projects. If your"
        " model is not a SpaCy project, use `bentoml.spacy.load()` instead."
    )


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    vocab: t.Union["Vocab", bool] = True,
    disable: t.Iterable[str] = util.SimpleFrozenList(),
    exclude: t.Iterable[str] = util.SimpleFrozenList(),
    config: t.Union[t.Dict[str, t.Any], "Config"] = util.SimpleFrozenDict(),
) -> "spacy.language.Language":
    bento_model = model_store.get(tag)

    try:
        projects_uri = bento_model.info.options.get("projects_uri")
        if projects_uri is not None:
            raise EnvironmentError(
                "Cannot use `bentoml.spacy.load()` to load SpaCy Projects. Use"
                " `bentoml.spacy.load_project()` instead."
            )
    except (AttributeError, TypeError):
        pass

    # Get required pip package
    try:
        required = bento_model.info.options.get("pip_package")
        if required:
            try:
                _ = importlib.import_module(required)
            except ModuleNotFoundError:
                try:
                    from spacy.cli.download import download

                    # Download model from SpaCy's model repository
                    download(required)
                except (SystemExit, Exception):
                    logger.warning(
                        f"{required} cannot be downloaded as pip package. If this"
                        f" is a custom pipeline there is nothing to worry about."
                        f" If this is a pretrained model provided by Explosion make"
                        f" sure that you save the correct package and model to BentoML"
                        f" via `bentoml.spacy.save()`"
                    )
    except (AttributeError, TypeError):
        pass

    # Check for additional requirements
    try:
        additional_requirements = bento_model.info.options.get(
            "additional_requirements"
        )
    except (AttributeError, TypeError):
        pass

    # Load the model from disk
    return util.load_model(
        bento_model.path, vocab=vocab, disable=disable, exclude=exclude, config=config
    )


def projects(
    save_name: str,
    tasks: str,
    name: t.Optional[str] = None,
    repo_or_store: t.Optional[str] = None,
    remotes_config: t.Optional[t.Dict[str, t.Dict[str, str]]] = None,
    *,
    branch: t.Optional[str] = None,
    sparse_checkout: bool = False,
    verbose: bool = True,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> bentoml.Model:
    """
    Save a SpaCy project to BentoML model store.

    Args:
        save_name: Name to save the project under
        tasks: SpaCy project task to perform (clone or pull)
        name: Name of the project template
        repo_or_store: Git repository URL or template store
        remotes_config: Configuration for remote project assets
        branch: Git branch to use
        sparse_checkout: Use Git sparse checkout
        verbose: Show verbose output
        metadata: Custom metadata

    Returns:
        A BentoML Model object representing the saved project

    Raises:
        BentoMLException: If task is not supported
    """
    from spacy.cli.project.pull import project_pull
    from spacy.cli.project.clone import project_clone

    if repo_or_store is None:
        repo_or_store = getattr(spacy.about, "__projects__")
    if branch is None:
        branch = getattr(spacy.about, "__projects_branch__")

    if tasks in PROJECTS_CMD_NOT_SUPPORTED:
        raise BentoMLException(
            """\
        BentoML only supports `clone` and `pull` for
         git and remote storage from SpaCy CLI to
         save into modelstore. Refers to
         https://spacy.io/api/cli#project for more
         information on SpaCy Projects.
        """
        )

    context = {
        "spacy_version": _spacy_version,
        "tasks": tasks,
    }

    save_dir = "project"

    options = {"projects_uri": repo_or_store, "target_path": save_dir}

    if tasks == "clone":
        if name is None:
            raise ValueError("`name` of the template is required to clone a project.")
        options["name"] = name

    with bentoml.models.create(
        save_name,
        module=sys.modules[__name__],
        options=options,
        context=context,
        metadata=metadata,
    ) as bento_model:
        output_path = os.path.join(bento_model.path, save_dir)
        os.makedirs(output_path, exist_ok=True)

        if tasks == "clone":
            project_clone(
                name,
                Path(output_path),
                repo=repo_or_store,
                branch=branch,
                sparse_checkout=sparse_checkout,
            )
        else:  # pull
            if remotes_config is None:
                raise ValueError(
                    """\
                    `remotes_config` is required to pull projects into BentoML modelstore.
                    Refer to https://spacy.io/usage/projects#remote for more information.
                    We will accept remotes as shown:
                    {
                        'remotes': {
                            'default':'s3://spacy-bucket',
                        }
                    }
                    """
                )

            with Path(output_path, "project.yml").open("w") as inf:
                yaml.dump(remotes_config, inf)

            for remote in remotes_config.get("remotes", {}):
                for url, res_path in project_pull(
                    Path(output_path), remote=remote, verbose=verbose
                ):
                    if url is not None:
                        logger.info(f"Pulled {res_path} from {repo_or_store}")

    return bento_model


class SpacyRunnable(bentoml.Runnable):
    """SpaCy model runnable for BentoML."""

    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, bento_model: bentoml.Model):
        """
        Initialize a SpacyRunnable.

        Args:
            bento_model: The BentoML model containing a SpaCy model
        """
        super().__init__()

        self._configure_device()

        # Load options
        self.vocab = True  # Default value
        self.disable = util.SimpleFrozenList()
        self.exclude = util.SimpleFrozenList()
        self.config = util.SimpleFrozenDict()

        self.model = load(bento_model.tag)

    def _configure_device(self):
        """Configure GPU or CPU device based on availability."""
        if self._has_gpu():
            backend = self._select_backend()

            if thinc_util.prefer_gpu(0):  # Use first GPU
                if backend == "pytorch":
                    thinc_backends.use_pytorch_for_gpu_memory()
                else:
                    thinc_backends.use_tensorflow_for_gpu_memory()

                thinc_util.require_gpu(0)
                thinc_backends.set_gpu_allocator(backend)
                thinc_util.set_active_gpu(0)
        else:
            thinc_util.require_cpu()

    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
                return True
            if tensorflow:
                # Try to access tensorflow GPU devices
                gpus = tensorflow.config.list_physical_devices("GPU")
                return len(gpus) > 0
        except (ImportError, AttributeError):
            pass
        return False

    def _select_backend(self) -> str:
        """Select backend for GPU memory allocation."""
        # Prefer PyTorch if available
        if torch and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "pytorch"
        return "tensorflow"

    def __call__(self, text: str) -> "Doc":
        """
        Process text with the SpaCy model.

        Args:
            text: Input text to process

        Returns:
            SpaCy Doc object
        """
        return self.model(text)

    def pipe(
        self,
        texts: t.Iterable[str],
        *,
        batch_size: t.Optional[int] = None,
        as_tuples: bool = False,
        **kwargs: t.Any,
    ) -> t.Iterable["Doc"]:
        """
        Process multiple texts with the SpaCy model.

        Args:
            texts: Iterable of input texts
            batch_size: Size of batches to process
            as_tuples: Whether the input and outputs are tuples
            **kwargs: Additional arguments for SpaCy pipe method

        Returns:
            Iterable of SpaCy Doc objects
        """
        return self.model.pipe(
            texts, as_tuples=as_tuples, batch_size=batch_size, **kwargs
        )


def get_runnable(bento_model: bentoml.Model) -> t.Type[SpacyRunnable]:
    """
    Get a SpacyRunnable for the given model.

    Args:
        bento_model: The BentoML model containing a SpaCy model

    Returns:
        SpacyRunnable class configured for the model
    """

    class BoundSpacyRunnable(SpacyRunnable):
        def __init__(self):
            super().__init__(bento_model)

    for method_name in ["__call__", "pipe"]:
        if method_name in bento_model.info.signatures:
            options = bento_model.info.signatures[method_name]

            # Get the method from SpacyRunnable
            method = getattr(SpacyRunnable, method_name)

            # Access attributes as dictionary or object properties safely
            try:
                if hasattr(options, "batchable"):
                    batchable = options.batchable
                elif hasattr(options, "get"):
                    batchable = options.get("batchable", False)
                else:
                    batchable = False

                if hasattr(options, "batch_dim"):
                    batch_dim = options.batch_dim
                elif hasattr(options, "get"):
                    batch_dim = options.get("batch_dim", 0)
                else:
                    batch_dim = 0

                if hasattr(options, "input_spec"):
                    input_spec = options.input_spec
                elif hasattr(options, "get"):
                    input_spec = options.get("input_spec", None)
                else:
                    input_spec = None

                if hasattr(options, "output_spec"):
                    output_spec = options.output_spec
                elif hasattr(options, "get"):
                    output_spec = options.get("output_spec", None)
                else:
                    output_spec = None
            except (AttributeError, TypeError):
                batchable = False
                batch_dim = 0
                input_spec = None
                output_spec = None

            # Add method to the bound runnable
            BoundSpacyRunnable.add_method(
                method,
                name=method_name,
                batchable=batchable,
                batch_dim=batch_dim,
                input_spec=input_spec,
                output_spec=output_spec,
            )

    return BoundSpacyRunnable
