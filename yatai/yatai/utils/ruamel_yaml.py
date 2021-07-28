# flake8: noqa
# pylint: skip-file
# Workaround for https://github.com/bentoml/BentoML/issues/984
# Issue 984 is largely fixed by enabling the conda `pip_interop_enabled` option,
# although this option is only available after conda version 4.6.0
# This import wrapper makes sure ruamel.yaml works properly when used with a docker base
# image that has installed conda version < 4.6.0
# See more details in https://github.com/bentoml/BentoML/pull/1012
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
