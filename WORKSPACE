# TODO: Migrate to bzlmod once 6.0.0 is released.
workspace(name = "com_github_bentoml_bentoml")

load("//rules:deps.bzl", "bentoml_dependencies")

bentoml_dependencies()

# NOTE: external users wish to use BentoML workspace setup
# should always be loaded in this order.
load("//rules:workspace0.bzl", "workspace0")

workspace0()

load("//rules:workspace1.bzl", "workspace1")

workspace1()

load("//rules:workspace2.bzl", "workspace2")

workspace2()

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    requirements = "//:bazel-requirements.lock.txt",
)

pip_parse(
    name = "tests",
    requirements = "//:bazel-tests-requirements.lock.txt",
)

load("@pypi//:requirements.bzl", pypi_deps = "install_deps")

pypi_deps()

load("//rules/py:deps.bzl", tests_deps = "install_deps")

tests_deps()
