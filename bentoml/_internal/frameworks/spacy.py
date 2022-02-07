import os
import sys
import typing as t
import logging
import importlib
from typing import TYPE_CHECKING
from pathlib import Path
from functools import partial
from distutils.dir_util import copy_tree

import yaml
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils import LazyLoader
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from ..bento.pip_pkg import split_requirement
from ..bento.pip_pkg import packages_distributions
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:  # pragma: no cover
    from spacy.vocab import Vocab
    from thinc.config import Config
    from spacy.tokens.doc import Doc

    from ..models import ModelStore

if sys.version_info >= (3, 8):
    from typing import Literal
else:  # pragma: no cover
    from typing_extensions import Literal

try:
    import spacy
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `spacy` is required to use with `bentoml.spacy`.
        Instruction: Refers to https://spacy.io/usage for more information.
        """
    )

MODULE_NAME = "bentoml.spacy"


util = LazyLoader("util", globals(), "spacy.util")
thinc_util = LazyLoader("thinc_util", globals(), "thinc.util")
thinc_backends = LazyLoader("thinc_backends", globals(), "thinc.backends")

torch = LazyLoader("torch", globals(), "torch")
tensorflow = LazyLoader("tensorflow", globals(), "tensorflow")

_TORCH_TF_WARNING = """\
It is recommended that if you want to run SpaCy with {framework}
 model that you also have `{package}` installed.
Refers to {link} for more information.

We also detected that you choose to run on GPUs, thus in order to utilize
 BentoML Runners features with GPUs you should also install `{package}` with
 CUDA support.
"""  # noqa

PROJECTS_CMD_NOT_SUPPORTED = [
    "assets",
    "document",
    "dvc",
    "push",
    "run",
]

logger = logging.getLogger(__name__)


@inject
def load_project(
    tag: Tag,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    if "projects_uri" in model.info.options:
        logger.warning(
            "We will only returns the path of projects saved under BentoML modelstore"
            " and leave projects interaction for users to decide what they want to do."
            " Refers to https://spacy.io/api/cli#project for more information."
        )
        return os.path.join(model.path, model.info.options["target_path"])
    raise BentoMLException(
        "Cannot use `bentoml.spacy.load_project()` to load non Spacy Projects. If your"
        " model is not a Spacy projects use `bentoml.spacy.load()` instead."
    )


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    vocab: t.Union["Vocab", bool] = True,  # type: ignore[reportUnknownParameterType]
    disable: t.Iterable[str] = util.SimpleFrozenList(),  # noqa
    exclude: t.Iterable[str] = util.SimpleFrozenList(),  # noqa
    config: t.Union[t.Dict[str, t.Any], "Config"] = util.SimpleFrozenDict(),  # noqa
) -> "spacy.language.Language":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        vocab (:code:`Union[spacy.vocab.Vocab, bool]`, `optional`, defaults to `True`):
            Optional vocab to pass in on initialization. If True, a new Vocab object will be created.
        disable (`Iterable[str]`, `optional`):
            Names of pipeline components to disable.
        exclude (`Iterable[str]`, `optional`):
            Names of pipeline components to exclude. Excluded
            components won't be loaded.
        config (:code:`Union[Dict[str, Any], spacy.Config]`, `optional`):
            Config overrides as nested dict or dict
            keyed by section values in dot notation.

    Returns:
        :obj:`spacy.language.Language`: an instance of :obj:`spacy.Language` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.spacy.load('custom_roberta')
    """
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    if "projects_uri" in model.info.options:
        raise BentoMLException(
            "Cannot use `bentoml.spacy.load()` to load Spacy Projects. Use"
            " `bentoml.spacy.load_project()` instead."
        )
    required = model.info.options["pip_package"]

    try:
        _ = importlib.import_module(required)
    except ModuleNotFoundError:
        try:
            from spacy.cli.download import download

            # TODO: move this to runner on startup hook
            download(required)
        except (SystemExit, Exception):  # pylint: disable=broad-except
            logger.warning(
                f"{required} cannot be downloaded as pip package. If this"
                " is a custom pipeline there is nothing to worry about."
                " If this is a pretrained model provided by Explosion make"
                " sure that you save the correct package and model to BentoML"
                " via `bentoml.spacy.save()`"
            )
    try:
        # check if pipeline has additional requirements then all related
        # pip package has been installed correctly.
        additional = model.info.options["additional_requirements"]
        not_existed = list()  # type: t.List[str]
        dists = packages_distributions()
        for module_name in additional:
            mod, _ = split_requirement(module_name)
            if mod not in dists:
                not_existed.append(module_name)
            if len(not_existed) > 0:
                raise MissingDependencyException(
                    f"`{','.join(not_existed)}` is required by `{tag}`."
                )
    except KeyError:
        pass
    return util.load_model(
        model.path, vocab=vocab, disable=disable, exclude=exclude, config=config
    )


@inject
def save(
    name: str,
    model: "spacy.language.Language",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`spacy.language.Language`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import spacy
        import bentoml.spacy
        nlp = spacy.load("en_core_web_trf")

        # custom training or layers here

        bentoml.spacy.save("spacy_roberta", nlp)
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "spacy",
        "pip_dependencies": [f"spacy=={get_pkg_version('spacy')}"],
    }
    _model = Model.create(
        name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )

    meta = model.meta
    pip_package = f"{meta['lang']}_{meta['name']}"
    _model.info.options = {"pip_package": pip_package}
    if "requirements" in meta:
        _model.info.options["additional_requirements"] = meta["requirements"]
    model.to_disk(_model.path)

    _model.save(model_store)
    return _model.tag


@inject
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
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Enables users to use :code:`spacy cli` and integrate SpaCy `Projects <https://spacy.io/usage/projects>`_ to BentoML.

    Args:
        save_name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        tasks (:code:`str`):
            Given SpaCy CLI tasks. Currently only support :code:`pull` and :code:`clone`
        repo_or_store(:code:`str`, `optional`, defaults to `None`):
            URL of Git repo or given S3 store containing project templates.
        model (`spacy.language.Language`):
            Instance of model to be saved
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        branch (:code:`str`, `optional`, defaults to `None`):
            The branch to clone from. If not specified, defaults to :code:`main` branch
        verbose (`bool`, `optional`, default to :code:`True`):
            Verbosely post all logs.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    .. warning::

       This is an **EXPERIMENTAL** API as it is subjected to change. We are also looking for feedback.

    Examples:

    .. code-block:: python

        import bentoml

        clone_tag = bentoml.spacy.projects(
            "test_spacy_project",
            "clone",
            name="integrations/huggingface_hub",
            repo_or_store="https://github.com/aarnphm/bentoml-spacy-projects-integration-tests",
        )
        project_path = bentoml.spacy.load_project(clone_tag)


        project_yml = {
            "remotes": {
                "default": "https://github.com/aarnphm/bentoml-spacy-projects-integration-tests/tree/v3/pipelines/tagger_parser_ud",
            }
        }
        pull_tag = bentoml.spacy.projects("test_pull", "pull", remotes_config=project_yml)
        project_path = bentoml.spacy.load_project(pull_tag)
    """  # noqa
    # EXPERIMENTAL: note that these functions are direct modified implementation
    # from spacy internal API. Subject to change, use with care!
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
    context: t.Dict[str, t.Any] = {
        "framework_name": "spacy",
        "pip_dependencies": [f"spacy=={get_pkg_version('spacy')}"],
        "tasks": tasks,
    }
    _model = Model.create(
        save_name,
        module=MODULE_NAME,
        options=None,
        context=context,
        metadata=metadata,
    )
    output_path = _model.path_of(SAVE_NAMESPACE)
    _model.info.options = {"projects_uri": repo_or_store, "target_path": SAVE_NAMESPACE}
    if tasks == "clone":
        # TODO: update check for master or main branch
        assert (
            name is not None
        ), "`name` of the template is required to clone a project."
        _model.info.options["name"] = name
        assert isinstance(repo_or_store, str) and isinstance(branch, str)
        project_clone(
            name,
            Path(output_path),
            repo=repo_or_store,
            branch=branch,
            sparse_checkout=sparse_checkout,
        )
        copy_tree(_model.path, output_path)
    else:
        # works with S3 bucket, haven't failed yet
        assert (
            remotes_config is not None
        ), """\
            `remotes_config` is required in order to pull projects into
             BentoML modelstore. Refers to
             https://spacy.io/usage/projects#remote
             for more information. We will accept remotes
             as shown:
             {
                'remotes':
                {
                    'default':'s3://spacy-bucket',
                }
            }
             """
        os.makedirs(output_path, exist_ok=True)
        with Path(output_path, "project.yml").open("w") as inf:
            yaml.dump(remotes_config, inf)
        for remote in remotes_config.get("remotes", {}):
            pull: "partial[t.Generator[t.Tuple[str, str], None, None]]" = partial(
                project_pull, remote=remote, verbose=verbose
            )
            for url, res_path in pull(Path(output_path)):
                if url is not None:  # pragma: no cover
                    logger.info(f"Pulled {res_path} from {repo_or_store}")
    _model.save(model_store)
    return _model.tag


class _SpacyRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        gpu_device_id: t.Optional[int],
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        vocab: t.Union["Vocab", bool],  # type: ignore[reportUnknownParameterType]
        disable: t.Iterable[str],
        exclude: t.Iterable[str],
        config: t.Union[t.Dict[str, t.Any], "Config"],
        as_tuples: t.Union[Literal[True], Literal[False]],
        batch_size: t.Optional[int],
        component_cfg: t.Optional[t.Dict[str, t.Dict[str, t.Any]]],
        backend_options: t.Optional[Literal["pytorch", "tensorflow"]] = "pytorch",
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        in_store_tag = model_store.get(tag).tag
        super().__init__(name, resource_quota, batch_options)
        self._tag = in_store_tag

        self._vocab: t.Union["Vocab", bool] = vocab
        self._disable = disable
        self._exclude = exclude
        self._config = config

        self._model_store = model_store
        self._backend_options = backend_options
        self._gpu_device_id = gpu_device_id

        if self._gpu_device_id is not None:
            if resource_quota is None:
                resource_quota = dict(gpus=self._gpu_device_id)
            else:
                resource_quota["gpus"] = self._gpu_device_id
        self._configure(backend_options)
        self._as_tuples = as_tuples
        self._batch_size = batch_size
        self._component_cfg = component_cfg
        super().__init__(str(tag), resource_quota, batch_options)

    def _configure(self, backend_options: t.Optional[str]) -> None:
        if self._gpu_device_id is not None and thinc_util.prefer_gpu(
            self._gpu_device_id
        ):  # pragma: no cover
            assert backend_options is not None
            if backend_options == "pytorch":
                thinc_backends.use_pytorch_for_gpu_memory()
            else:
                thinc_backends.use_tensorflow_for_gpu_memory()
            thinc_util.require_gpu(self._gpu_device_id)
            thinc_backends.set_gpu_allocator(backend_options)
            thinc_util.set_active_gpu(self._gpu_device_id)
        else:
            thinc_util.require_cpu()

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    def _get_pytorch_gpu_count(self) -> t.Optional[int]:
        assert self._backend_options == "pytorch"
        devs = getattr(torch, "cuda").device_count()
        if devs == 0:
            logger.warning("Installation of Torch is not CUDA-enabled.")
            logger.warning(
                _TORCH_TF_WARNING.format(
                    framework="PyTorch",
                    package="torch",
                    link="https://pytorch.org/get-started/locally/",
                )
            )
            return None
        return devs

    def _get_tensorflow_gpu_count(self) -> t.Optional[int]:
        assert self._backend_options == "tensorflow"
        from tensorflow.python.client import (
            device_lib,  # type: ignore; pylint: disable=E0611
        )

        try:
            return len(
                [
                    x
                    for x in getattr(device_lib, "list_local_devices")()
                    if getattr(x, "device_type") == "GPU"
                ]
            )
        except (AttributeError, Exception):  # pylint: disable=broad-except
            logger.warning(
                _TORCH_TF_WARNING.format(
                    framework="Tensorflow 2.x",
                    package="tensorflow",
                    link="https://www.tensorflow.org/install/gpu",
                )
            )
            return None

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            if self._backend_options == "pytorch":
                num_devices = self._get_pytorch_gpu_count()
            else:
                num_devices = self._get_tensorflow_gpu_count()
            return num_devices if num_devices is not None else 1
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:
        self._model = load(
            self._tag,
            model_store=self._model_store,
            vocab=self._vocab,
            exclude=self._exclude,
            disable=self._disable,
            config=self._config,
        )

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        input_data: t.Union[t.Iterable[t.Tuple[t.Union[str, "Doc"], t.Any]], t.Union[str, "Doc"]],  # type: ignore[reportUnknownParameterType] # noqa: LN001
    ) -> t.Union[t.Iterator["Doc"], t.Iterator[t.Tuple["Doc", t.Any]]]:  # type: ignore[reportUnknownParameterType] # noqa: LN001
        return self._model.pipe(  # type: ignore[reportGeneralTypeIssues]
            input_data,  # type: ignore[reportGeneralTypeIssues]
            as_tuples=self._as_tuples,
            batch_size=self._batch_size,
            disable=self._disable,
            component_cfg=self._component_cfg,
            n_process=self.num_replica,
        )


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    gpu_device_id: t.Optional[int] = None,
    backend_options: t.Optional[Literal["pytorch", "tensorflow"]] = None,
    name: t.Optional[str] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    vocab: t.Union["Vocab", bool] = True,  # type: ignore[reportUnknownParameterType]
    disable: t.Iterable[str] = util.SimpleFrozenList(),  # noqa
    exclude: t.Iterable[str] = util.SimpleFrozenList(),  # noqa
    config: t.Union[t.Dict[str, t.Any], "Config"] = util.SimpleFrozenDict(),  # noqa
    as_tuples: t.Union[Literal[True], Literal[False]] = False,
    batch_size: t.Optional[int] = None,
    component_cfg: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_SpacyRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.spacy.load_runner` implements a Runner class that
    wrap around :obj:`spacy.language.Language` model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore..
        gpu_device_id (`int`, `optional`, defaults to `None`):
            GPU device ID.
        backend_options (`Literal['pytorch', 'tensorflow'], `optional`, defaults to `None`):
            Backend options for Thinc. Either PyTorch or Tensorflow.
        resource_quota (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        vocab (:code:`Union[spacy.vocab.Vocab, bool]`, `optional`, defaults to `True`):
            Optional vocab to pass in on initialization. If True, a new Vocab object will be created.
        disable (`Iterable[str]`, `optional`):
            Names of pipeline components to disable.
        exclude (`Iterable[str]`, `optional`):
            Names of pipeline components to exclude. Excluded
            components won't be loaded.
        config (:code:`Union[Dict[str, Any], spacy.Config]`, `optional`):
            Config overrides as nested dict or dict
            keyed by section values in dot notation.
        as_tuples (`Literal[False, True]`, `optional`, defaults to `False`):
            If set to True, inputs should be a sequence of
            (text, context) tuples. Output will then be a sequence of
            (doc, context) tuples.
        batch_size (`int`, `optional`, defaults to `None`):
            The number of texts to buffer.
        component_cfg (:code:`Dict[str, :code:`Dict[str, Any]]`, `optional`, defaults to `None`):
            An optional dictionary with extra keyword
            arguments for specific components.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for the target :mod:`bentoml.sklearn` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.sklearn.load_runner("my_model:latest")
        runner.run([[1,2,3,4]])
    """
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _SpacyRunner(
        tag=tag,
        gpu_device_id=gpu_device_id,
        backend_options=backend_options,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        vocab=vocab,
        disable=disable,
        exclude=exclude,
        config=config,
        as_tuples=as_tuples,
        batch_size=batch_size,
        component_cfg=component_cfg,
        model_store=model_store,
    )
