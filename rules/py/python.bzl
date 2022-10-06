load("@aspect_rules_py//py:defs.bzl", _pytest_main = "py_pytest_main")
load("@rules_python//python:pip.bzl", _compile_pip_requirements = "compile_pip_requirements")
load(":deps.bzl", "requirement")

FRAMEWORKS = [
    "catboost",
    "onnx",
    "picklable_model",
    "torchscript",
    "sklearn",
    "xgboost",
]

# NOTE: the following are frameworks with general tests and additional unit tests.
FRAMEWORKS_GENERAL_TESTS = [
    # transformers require extra deps from tensorflow and pytorch.
    "transformers",
    # the rest with additional general framework tests.
    # XXX: to sync with tests/integration/frameworks/BUILD#L47
    "fastai",
    "pytorch",
    "tensorflow",
]

FRAMEWORKS_UNIVERSAL = FRAMEWORKS + FRAMEWORKS_GENERAL_TESTS

# XXX: mlflow has its own testset. so it will be added separatedly in src/BUILD.bazel
# XXX: lightgbm can only be ran on Linux, and it is also added separately in tests/integration/frameworks/BUILD.
# XXX: pytorch_lightning depends on gRPC which needs to build manually on M1.
SUPPORTED_FRAMEWORKS_TARGETS = FRAMEWORKS_UNIVERSAL + ["lightgbm", "mlflow", "pytorch_lightning", "keras"]

def py_test(name, args = [], data = [], **kwargs):
    """A py_test macro that use pytest + rules_python's 'py_test' to run BentoML tests.

    This rule will create a target <name>, which will glob all 'srcs' that match 'test_<name>.py'.

    Note that if this rule will create a ":__test__" rule if given rule doesn't exist in current DAG.

    Args:
      name: The name of the test suite.
      **kwargs: Additional arguments to pass to py_test.
    """

    # __test__ is a special attribute from py_pytest_main
    if "__test__" not in native.existing_rules():
        _pytest_main(name = "__test__")

    srcs = kwargs.pop("srcs", [])
    deps = kwargs.pop("deps", [])
    imports = kwargs.pop("imports", [])

    if not srcs:
        srcs += ["test_{}.py".format(name)]

    native.py_test(
        name = name,
        srcs = [":__test__"] + srcs,
        main = ":__test__.py",
        args = ["-rfEX", "-vvv", "-p", "rules.py.codecoverage"] + args,
        python_version = "PY3",
        env = {
            # NOTE: Set the following envvar to build the wheel.
            "BENTOML_BUNDLE_LOCAL_BUILD": "True",
            "SETUPTOOLS_USE_DISTUTILS": "stdlib",
        },
        deps = [
            ":__test__",
            "//:sdk",
            "//:cli",
            "//rules/py:codecov",
            requirement("pytest"),
            requirement("pytest-xdist"),
            requirement("pytest-asyncio"),
            requirement("setuptools-scm"),
            requirement("build"),
            requirement("virtualenv"),
        ] + deps,
        imports = imports,
        data = [
            "//:pyproject",
            "//src/bentoml:srcs_files",
        ] + data,
        legacy_create_init = 0,
        **kwargs
    )

def framework_py_test(name, srcs = [], data = [], args = [], **kwargs):
    """An extension of py_test macro to run general framework tests."""
    if name not in SUPPORTED_FRAMEWORKS_TARGETS:
        fail("'{}' is not a supported target in {}.".format(name, SUPPORTED_FRAMEWORKS_TARGETS))

    deps = kwargs.pop("deps", [])
    args = kwargs.pop("args", [])

    deps += ["//tests/integration/frameworks:test_frameworks", "//tests/integration/frameworks/models:{}_test_lib".format(name)]

    py_test(
        name = name + "_test",
        deps = deps,
        srcs = srcs + ["test_frameworks.py"],
        args = args,
        data = data + ["//tests/integration/frameworks:conftest"],
        **kwargs
    )

def framework_py_library(name, deps = [], data = [], **kwargs):
    """An extension of py_library for general framework tests module"""
    if "visibility" in kwargs:
        fail("visibility will be set automatically, remove 'visibility = {}'".format(kwargs.get("visibility")))

    srcs = kwargs.pop("srcs", [])
    if not srcs:
        srcs = ["{}.py".format(name)]

    native.py_library(
        name = name + "_test_lib",
        srcs = srcs,
        deps = deps + ["//tests/integration/frameworks:models", "//src/bentoml:{}".format(name)],
        data = data + ["//tests/integration/frameworks:conftest"],
        visibility = ["//:__subpackages__"],
        **kwargs
    )
