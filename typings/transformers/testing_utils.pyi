

import contextlib
import os
import unittest
from typing import Iterator, Union

from .file_utils import is_tf_available, is_torch_available

SMALL_MODEL_IDENTIFIER = ...
DUMMY_UNKWOWN_IDENTIFIER = ...
DUMMY_DIFF_TOKENIZER_IDENTIFIER = ...
USER = ...
PASS = ...
ENDPOINT_STAGING = ...
def parse_flag_from_env(key, default=...):
    ...

def parse_int_from_env(key, default=...):
    ...

_run_slow_tests = ...
_run_pt_tf_cross_tests = ...
_run_pt_flax_cross_tests = ...
_run_custom_tokenizers = ...
_run_staging = ...
_run_pipeline_tests = ...
_run_git_lfs_tests = ...
_tf_gpu_memory_limit = ...
def is_pt_tf_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and TensorFlow.

    PT+TF tests are skipped by default and we can run only them by setting RUN_PT_TF_CROSS_TESTS environment variable
    to a truthy value and selecting the is_pt_tf_cross_test pytest mark.

    """
    ...

def is_pt_flax_cross_test(test_case):
    """
    Decorator marking a test as a test that control interactions between PyTorch and Flax

    PT+FLAX tests are skipped by default and we can run only them by setting RUN_PT_FLAX_CROSS_TESTS environment
    variable to a truthy value and selecting the is_pt_flax_cross_test pytest mark.

    """
    ...

def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test.

    Pipeline tests are skipped by default and we can run only them by setting RUN_PIPELINE_TESTS environment variable
    to a truthy value and selecting the is_pipeline_test pytest mark.

    """
    ...

def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    ...

def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    ...

def tooslow(test_case):
    """
    Decorator marking a test as too slow.

    Slow tests are skipped while they're in the process of being fixed. No test should stay tagged as "tooslow" as
    these will not be tested by the CI.

    """
    ...

def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped by default. Set the RUN_CUSTOM_TOKENIZERS
    environment variable to a truthy value to run them.
    """
    ...

def require_git_lfs(test_case):
    """
    Decorator marking a test that requires git-lfs.

    git-lfs requires additional dependencies, and tests are skipped by default. Set the RUN_GIT_LFS_TESTS environment
    variable to a truthy value to run them.
    """
    ...

def require_rjieba(test_case):
    """
    Decorator marking a test that requires rjieba. These tests are skipped when rjieba isn't installed.
    """
    ...

def require_keras2onnx(test_case):
    ...

def require_onnx(test_case):
    ...

def require_timm(test_case):
    """
    Decorator marking a test that requires Timm.

    These tests are skipped when Timm isn't installed.

    """
    ...

def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    ...

def require_torch_scatter(test_case):
    """
    Decorator marking a test that requires PyTorch scatter.

    These tests are skipped when PyTorch scatter isn't installed.

    """
    ...

def require_torchaudio(test_case):
    """
    Decorator marking a test that requires torchaudio. These tests are skipped when torchaudio isn't installed.
    """
    ...

def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow. These tests are skipped when TensorFlow isn't installed.
    """
    ...

def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax. These tests are skipped when one / both are not installed
    """
    ...

def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    ...

def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers. These tests are skipped when 🤗 Tokenizers isn't installed.
    """
    ...

def require_pandas(test_case):
    """
    Decorator marking a test that requires pandas. These tests are skipped when pandas isn't installed.
    """
    ...

def require_scatter(test_case):
    """
    Decorator marking a test that requires PyTorch Scatter. These tests are skipped when PyTorch Scatter isn't
    installed.
    """
    ...

def require_vision(test_case):
    """
    Decorator marking a test that requires the vision dependencies. These tests are skipped when torchaudio isn't
    installed.
    """
    ...

def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch). These tests are skipped on a machine without
    multiple GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    ...

def require_torch_non_multi_gpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    ...

def require_torch_up_to_2_gpus(test_case):
    """
    Decorator marking a test that requires 0 or 1 or 2 GPU setup (in PyTorch).
    """
    ...

def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    ...

if is_torch_available():
    torch_device = ...
else:
    torch_device = ...
if is_tf_available():
    ...
def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    ...

def require_datasets(test_case):
    """Decorator marking a test that requires datasets."""
    ...

def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    ...

def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    ...

def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    ...

def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile

    These tests are skipped when soundfile isn't installed.

    """
    ...

def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    ...

def get_gpu_count():
    """
    Return the number of available gpus (regardless of whether torch or tf is used)
    """
    ...

def get_tests_dir(append_path=...):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    ...

def apply_print_resets(buf):
    ...

def assert_screenout(out, what):
    ...

class CaptureStd:
    """
    Context manager to capture:

        - stdout, clean it up and make it available via obj.out
        - stderr, and make it available via obj.err

        init arguments:

        - out - capture stdout: True/False, default True
        - err - capture stdout: True/False, default True

        Examples::

            with CaptureStdout() as cs:
                print("Secret message")
            print(f"captured: {cs.out}")

            import sys
            with CaptureStderr() as cs:
                print("Warning: ", file=sys.stderr)
            print(f"captured: {cs.err}")

            # to capture just one of the streams, but not the other
            with CaptureStd(err=False) as cs:
                print("Secret message")
            print(f"captured: {cs.out}")
            # but best use the stream-specific subclasses

    """
    def __init__(self, out=..., err=...) -> None:
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *exc):
        ...
    
    def __repr__(self):
        ...
    


class CaptureStdout(CaptureStd):
    """Same as CaptureStd but captures only stdout"""
    def __init__(self) -> None:
        ...
    


class CaptureStderr(CaptureStd):
    """Same as CaptureStd but captures only stderr"""
    def __init__(self) -> None:
        ...
    


class CaptureLogger:
    """
    Context manager to capture `logging` streams

    Args:

    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example::

        >>> from transformers import logging
        >>> from transformers.testing_utils import CaptureLogger

        >>> msg = "Testing 1, 2, 3"
        >>> logging.set_verbosity_info()
        >>> logger = logging.get_logger("transformers.models.bart.tokenization_bart")
        >>> with CaptureLogger(logger) as cl:
        ...     logger.info(msg)
        >>> assert cl.out, msg+"\n"
    """
    def __init__(self, logger) -> None:
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *exc):
        ...
    
    def __repr__(self):
        ...
    


@contextlib.contextmanager
def LoggingLevel(level):
    """
    This is a context manager to temporarily change transformers modules logging level to the desired value and have it
    restored to the original setting at the end of the scope.

    For example ::

        with LoggingLevel(logging.INFO):
            AutoModel.from_pretrained("gpt2") # calls logger.info() several times

    """
    ...

@contextlib.contextmanager
def ExtendSysPath(path: Union[str, os.PathLike]) -> Iterator[None]:
    """
    Temporary add given path to `sys.path`.

    Usage ::

       with ExtendSysPath('/path/to/dir'):
           mymodule = importlib.import_module('mymodule')

    """
    ...

class TestCasePlus(unittest.TestCase):
    """
    This class extends `unittest.TestCase` with additional features.

    Feature 1: A set of fully resolved important file and dir path accessors.

    In tests often we need to know where things are relative to the current test file, and it's not trivial since the
    test could be invoked from more than one directory or could reside in sub-directories with different depths. This
    class solves this problem by sorting out all the basic paths and provides easy accessors to them:

    * ``pathlib`` objects (all fully resolved):

       - ``test_file_path`` - the current test file path (=``__file__``)
       - ``test_file_dir`` - the directory containing the current test file
       - ``tests_dir`` - the directory of the ``tests`` test suite
       - ``examples_dir`` - the directory of the ``examples`` test suite
       - ``repo_root_dir`` - the directory of the repository
       - ``src_dir`` - the directory of ``src`` (i.e. where the ``transformers`` sub-dir resides)

    * stringified paths---same as above but these return paths as strings, rather than ``pathlib`` objects:

       - ``test_file_path_str``
       - ``test_file_dir_str``
       - ``tests_dir_str``
       - ``examples_dir_str``
       - ``repo_root_dir_str``
       - ``src_dir_str``

    Feature 2: Flexible auto-removable temporary dirs which are guaranteed to get removed at the end of test.

    1. Create a unique temporary dir:

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir()

    ``tmp_dir`` will contain the path to the created temporary dir. It will be automatically removed at the end of the
    test.


    2. Create a temporary dir of my choice, ensure it's empty before the test starts and don't
    empty it after the test.

    ::

        def test_whatever(self):
            tmp_dir = self.get_auto_remove_tmp_dir("./xxx")

    This is useful for debug when you want to monitor a specific directory and want to make sure the previous tests
    didn't leave any data in there.

    3. You can override the first two options by directly overriding the ``before`` and ``after`` args, leading to the
       following behavior:

    ``before=True``: the temporary dir will always be cleared at the beginning of the test.

    ``before=False``: if the temporary dir already existed, any existing files will remain there.

    ``after=True``: the temporary dir will always be deleted at the end of the test.

    ``after=False``: the temporary dir will always be left intact at the end of the test.

    Note 1: In order to run the equivalent of ``rm -r`` safely, only subdirs of the project repository checkout are
    allowed if an explicit ``tmp_dir`` is used, so that by mistake no ``/tmp`` or similar important part of the
    filesystem will get nuked. i.e. please always pass paths that start with ``./``

    Note 2: Each test can register multiple temporary dirs and they all will get auto-removed, unless requested
    otherwise.

    Feature 3: Get a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` specific to the current test suite.
    This is useful for invoking external programs from the test suite - e.g. distributed training.


    ::
        def test_whatever(self):
            env = self.get_env()

    """
    def setUp(self):
        ...
    
    @property
    def test_file_path(self):
        ...
    
    @property
    def test_file_path_str(self):
        ...
    
    @property
    def test_file_dir(self):
        ...
    
    @property
    def test_file_dir_str(self):
        ...
    
    @property
    def tests_dir(self):
        ...
    
    @property
    def tests_dir_str(self):
        ...
    
    @property
    def examples_dir(self):
        ...
    
    @property
    def examples_dir_str(self):
        ...
    
    @property
    def repo_root_dir(self):
        ...
    
    @property
    def repo_root_dir_str(self):
        ...
    
    @property
    def src_dir(self):
        ...
    
    @property
    def src_dir_str(self):
        ...
    
    def get_env(self):
        """
        Return a copy of the ``os.environ`` object that sets up ``PYTHONPATH`` correctly, depending on the test suite
        it's invoked from. This is useful for invoking external programs from the test suite - e.g. distributed
        training.

        It always inserts ``./src`` first, then ``./tests`` or ``./examples`` depending on the test suite type and
        finally the preset ``PYTHONPATH`` if any (all full resolved paths).

        """
        ...
    
    def get_auto_remove_tmp_dir(self, tmp_dir=..., before=..., after=...):
        """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                if :obj:`None`:

                   - a unique temporary path will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=True`` if ``after`` is :obj:`None`
                else:

                   - :obj:`tmp_dir` will be created
                   - sets ``before=True`` if ``before`` is :obj:`None`
                   - sets ``after=False`` if ``after`` is :obj:`None`
            before (:obj:`bool`, `optional`):
                If :obj:`True` and the :obj:`tmp_dir` already exists, make sure to empty it right away if :obj:`False`
                and the :obj:`tmp_dir` already exists, any existing files will remain there.
            after (:obj:`bool`, `optional`):
                If :obj:`True`, delete the :obj:`tmp_dir` at the end of the test if :obj:`False`, leave the
                :obj:`tmp_dir` and its contents intact at the end of the test.

        Returns:
            tmp_dir(:obj:`string`): either the same value as passed via `tmp_dir` or the path to the auto-selected tmp
            dir
        """
        ...
    
    def tearDown(self):
        ...
    


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this ::

    @mockenv(RUN_SLOW=True, USE_TF=False)
    def test_something():
        run_slow = os.getenv("RUN_SLOW", False)
        use_tf = os.getenv("USE_TF", False)

    """
    ...

@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place. Similar to mockenv

    The ``os.environ`` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.
    """
    ...

pytest_opt_registered = ...
def pytest_addoption_shared(parser):
    """
    This function is to be called from `conftest.py` via `pytest_addoption` wrapper that has to be defined there.

    It allows loading both `conftest.py` files at once without causing a failure due to adding the same `pytest`
    option.

    """
    ...

def pytest_terminal_summary_main(tr, id):
    """
    Generate multiple reports at the end of test suite run - each report goes into a dedicated file in the current
    directory. The report files are prefixed with the test suite name.

    This function emulates --duration and -rA pytest arguments.

    This function is to be called from `conftest.py` via `pytest_terminal_summary` wrapper that has to be defined
    there.

    Args:

    - tr: `terminalreporter` passed from `conftest.py`
    - id: unique id like `tests` or `examples` that will be incorporated into the final reports filenames - this is
      needed as some jobs have multiple runs of pytest, so we can't have them overwrite each other.

    NB: this functions taps into a private _pytest API and while unlikely, it could break should
    pytest do internal changes - also it calls default internal methods of terminalreporter which
    can be hijacked by various `pytest-` plugins and interfere.

    """
    ...

class _RunOutput:
    def __init__(self, returncode, stdout, stderr) -> None:
        ...
    


def execute_subprocess_async(cmd, env=..., stdin=..., timeout=..., quiet=..., echo=...) -> _RunOutput:
    ...

def pytest_xdist_worker_id():
    """
    Returns an int value of worker's numerical id under ``pytest-xdist``'s concurrent workers ``pytest -n N`` regime,
    or 0 if ``-n 1`` or ``pytest-xdist`` isn't being used.
    """
    ...

def get_torch_dist_unique_port():
    """
    Returns a port number that can be fed to ``torch.distributed.launch``'s ``--master_port`` argument.

    Under ``pytest-xdist`` it adds a delta number based on a worker id so that concurrent tests don't try to use the
    same port at once.
    """
    ...

def nested_simplify(obj, decimals=...):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    ...

