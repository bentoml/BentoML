import importlib
import logging
import os
import sys
import typing as t
from hashlib import sha256
from pathlib import Path

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.environment.pip_pkg import packages_distributions, split_requirement
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from _internal.models.store import ModelStore
    from spacy import Config, Vocab
    from spacy.tokens.doc import Doc

try:
    import spacy
    import spacy.cli
    from spacy.util import SimpleFrozenDict, SimpleFrozenList
    from thinc.api import (
        prefer_gpu,
        require_cpu,
        require_gpu,
        set_active_gpu,
        set_gpu_allocator,
        use_pytorch_for_gpu_memory,
        use_tensorflow_for_gpu_memory,
    )
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `spacy` is required to use with `bentoml.spacy`.
        Instruction: Refers to https://spacy.io/usage for more information.
        """
    )

_check_compat = spacy.__version__.startswith("3")
if not _check_compat:  # pragma: no cover
    # TODO: supports spacy 2.x?
    raise EnvironmentError(
        "BentoML will only provide supports for spacy 3.x and above"
        " as we can provide more supports for Spacy new design. Currently"
        f" detected spacy to have version {spacy.__version__}"
    )

try:  # pragma: no cover
    import tensorflow
    import torch
except ImportError:  # pragma: no cover
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

DEFAULT_SPACY_PROJECTS_REPO = spacy.about.__projects__
DEFAULT_SPACY_PROJECTS_BRANCH = "v3"

logger = logging.getLogger(__name__)


def _uri_to_filename(uri: t.Optional[str]) -> str:  # pragma: no cover
    if uri is None:
        uri = DEFAULT_SPACY_PROJECTS_REPO
    return f"spc{sha256(uri.encode('utf-8')).hexdigest()}"


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    vocab: t.Union["Vocab", bool] = True,
    disable: t.Iterable[str] = SimpleFrozenList(),
    exclude: t.Iterable[str] = SimpleFrozenList(),
    config: t.Union[t.Dict[str, t.Any], "Config"] = SimpleFrozenDict(),
) -> "spacy.language.Language":
    model_info = model_store.get(tag)
    if "projects_uri" in model_info.options:
        raise EnvironmentError(
            """\
        `bentoml.spacy.load()` is not designed to load or run a projects.
        We will leave projects interaction for users to decide what they want to do.
        Refers to https://spacy.io/api/cli#project for more information.
        """
        )
    required = model_info.options["pip_package"]
    try:
        _ = importlib.import_module(required)
    except ModuleNotFoundError:
        try:
            spacy.cli.download(required)
        except BaseException:
            logger.warning(
                f"{required} cannot be downloaded as pip package. If this"
                f" is a custom pipeline there is nothing to worry about."
                f" If this is a pretrained model provided by Explosion make"
                f" sure that you save the correct package and model to BentoML"
                f" via `bentoml.spacy.save()`"
            )
            pass

    try:
        # check if pipeline has additional requirements then all related
        # pip package has been installed correctly.
        additional = model_info.options["additional_requirements"]
        not_existed = list()
        dists = packages_distributions()
        for module_name in additional:
            mod, spec = split_requirement(module_name)
            if mod not in dists:
                not_existed.append(module_name)
            if len(not_existed) > 0:
                raise MissingDependencyException(
                    f"`{','.join(not_existed)}` is required by `{tag}`."
                )
    except KeyError:
        pass
    return spacy.util.load_model(
        model_info.path, vocab=vocab, disable=disable, exclude=exclude, config=config
    )


@inject
def save(
    name: str,
    model: "spacy.language.Language",
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`spacy.language.Language`):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import spacy
        import bentoml.spacy
        nlp = spacy.load("en_core_web_trf")

        # custom training or layers here

        bentoml.spacy.save("spacy_roberta", nlp)
    """  # noqa
    context = {"spacy": spacy.__version__}
    with model_store.register(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        meta = model.meta  # type: t.Dict[str, t.Any]
        ctx.options = {
            "pip_package": f"{meta['lang']}_{meta['name']}",
        }
        if "requirements" in meta:
            ctx.options["additional_requirements"] = meta["requirements"]
        model.to_disk(ctx.path)
        return ctx.tag  # type: ignore[no-any-return]


# TODO: save local projects
# bentoml.spacy.projects('save', /path/to/local/spacy/project)
@inject
def projects(
    tasks: str,
    name: t.Optional[str] = None,
    repo_or_store: str = DEFAULT_SPACY_PROJECTS_REPO,
    *,
    branch: str = DEFAULT_SPACY_PROJECTS_BRANCH,
    sparse_checkout: bool = False,
    verbose: bool = True,
    remote: str = "defaults",
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Tuple[str, "Path"]:
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
        "spacy": spacy.__version__,
        "tasks": tasks,
    }
    with model_store.register(
        _uri_to_filename(repo_or_store),
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        output_path = Path(ctx.path, SAVE_NAMESPACE)
        ctx.options = {"projects_uri": repo_or_store}
        if tasks == "clone":
            # TODO: update check for master or main branch
            assert (
                name is not None
            ), "`name` of the template is required to clone a project."
            ctx.options["name"] = name
            spacy.cli.project_clone(
                name,
                output_path,
                repo=repo_or_store,
                branch=branch,
                sparse_checkout=sparse_checkout,
            )
        else:
            current_file = os.path.realpath(sys.argv[0])
            assert "project.yml" in [
                i.name for i in Path(current_file).parent.iterdir()
            ], (
                "`project.yml` is required "
                f"in {str(Path(current_file).parent.resolve())}"
                " to pull projects into"
                " BentoML modelstore. Refers to"
                " https://spacy.io/usage/projects#remote"
                " for more information."
            )
            for url, output_path in spacy.cli.project_pull(
                output_path, remote=remote, verbose=verbose
            ):
                if url is not None:
                    logger.debug(f"Pulled {output_path} from {repo_or_store}")

        return ctx.tag, output_path


class _SpacyRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        gpu_device_id: t.Optional[int],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        vocab: t.Union["Vocab", bool],
        disable: t.Iterable[str],
        exclude: t.Iterable[str],
        config: t.Union[t.Dict[str, t.Any], "Config"],
        backend_options: t.Optional[t.Literal["pytorch", "tensorflow"]] = "pytorch",
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        self._vocab = vocab
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
        super().__init__(tag, resource_quota, batch_options)

    def _configure(self, backend_options: t.Optional[str]) -> None:
        if self._gpu_device_id is not None and prefer_gpu(self._gpu_device_id):
            assert backend_options is not None
            if backend_options == "pytorch":
                use_pytorch_for_gpu_memory()
            else:
                use_tensorflow_for_gpu_memory()
            require_gpu(self._gpu_device_id)
            set_gpu_allocator(backend_options)
            set_active_gpu(self._gpu_device_id)
        else:
            require_cpu()

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_store.get(self.name).tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        if self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if self.resource_quota.on_gpu:
            if self._backend_options == "pytorch":
                if hasattr(torch, "cuda"):
                    return torch.cuda.device_count()
                else:
                    logger.warning(
                        _TORCH_TF_WARNING.format(
                            framework="PyTorch",
                            package="torch",
                            link="https://pytorch.org/get-started/locally/",
                        )
                    )
            else:
                if tensorflow is not None:
                    from tensorflow.python.client import device_lib

                    return len(
                        [
                            x
                            for x in device_lib.list_local_devices()
                            if x.device_type == "GPU"
                        ]
                    )
                else:
                    logger.warning(
                        _TORCH_TF_WARNING.format(
                            framework="Tensorflow 2.x",
                            package="tensorflow",
                            link="https://www.tensorflow.org/install/gpu",
                        )
                    )
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(
            self.name,
            model_store=self._model_store,
            vocab=self._vocab,
            exclude=self._exclude,
            disable=self._disable,
            config=self._config,
        )

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self,
        inputs: t.Iterable[t.Tuple[str, t.Any]],
        as_tuples: bool = False,
        batch_size: t.Optional[int] = None,
        disable: t.Iterable[str] = SimpleFrozenList(),
        component_cfg: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None,
    ) -> t.Iterator[t.Tuple[t.Union["Doc", "Doc"], t.Any]]:
        return self._model.pipe(
            inputs,
            as_tuples=as_tuples,
            batch_size=batch_size,
            disable=disable,
            component_cfg=component_cfg,
            n_process=self.num_replica,
        )


@inject
def load_runner(
    tag: str,
    *,
    gpu_device_id: t.Optional[int] = None,
    backend_options: t.Optional[t.Literal["pytorch", "tensorflow"]] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    vocab: t.Union["Vocab", bool] = True,
    disable: t.Iterable[str] = SimpleFrozenList(),
    exclude: t.Iterable[str] = SimpleFrozenList(),
    config: t.Union[t.Dict[str, t.Any], "Config"] = SimpleFrozenDict(),
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_SpacyRunner":
    return _SpacyRunner(
        tag=tag,
        gpu_device_id=gpu_device_id,
        backend_options=backend_options,
        resource_quota=resource_quota,
        batch_options=batch_options,
        vocab=vocab,
        disable=disable,
        exclude=exclude,
        config=config,
        model_store=model_store,
    )
