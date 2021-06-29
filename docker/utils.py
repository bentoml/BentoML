#  Copyright (c) 2021 Atalaya Tech, Inc.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==========================================================================

import errno
import logging
import os
from pathlib import Path
from typing import Callable, MutableMapping, Iterable, Dict, List

from absl import flags

__all__ = (
    'FLAGS',
    'cached_property',
    'ColoredFormatter',
    'mkdir_p',
    'flatten',
    'mapfunc',
    'maxkeys',
    'walk',
)

# defined global vars.
FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    "push_to_hub",
    False,
    "Whether to upload images to given registries.",
    short_name="pth",
)
flags.DEFINE_integer("timeout", 3600, "Timeout for docker build", short_name="t")
# CLI-related
flags.DEFINE_boolean(
    "dump_metadata",
    False,
    "Dump all tags metadata to file and stops if specified.",
    short_name="dm",
)
flags.DEFINE_boolean(
    "dry_run",
    False,
    "Whether to dry run. This won't create Dockerfile.",
    short_name="dr",
)
flags.DEFINE_boolean("overwrite", False, "Overwrite built images.", short_name="o")
flags.DEFINE_boolean(
    "stop_at_generate", False, "Stop at generating Dockerfile.", short_name="sag"
)
# directory and files
flags.DEFINE_string(
    "dockerfile_dir",
    "./generated",
    "path to generated Dockerfile. Existing files will be deleted with new Dockerfiles",
    short_name="dd",
)
flags.DEFINE_string("manifest_file", "./manifest.yml", "Manifest file", short_name="mf")

flags.DEFINE_string(
    "bentoml_version", None, "BentoML release version", required=True, short_name="bv"
)
flags.DEFINE_string("cuda_version", "11.3.1", "Define CUDA version", short_name="cv")

flags.DEFINE_multi_string(
    "python_version",
    [],
    "OPTIONAL: Python version to build Docker images (useful when developing).",
    short_name="pv",
)


class cached_property(property):
    """Direct ports from bentoml/utils/__init__.py"""

    class _Missing(object):
        def __repr__(self):
            return "no value"

        def __reduce__(self):
            return "_missing"

    _missing = _Missing()

    def __init__(self, func, name=None, doc=None):
        super().__init__(doc)
        self.__name__ = name or func.__name__
        self.__doc__ = doc or func.__doc__
        self.__module__ = func.__module__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, types=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    reset = "\x1b[0m"
    _format = "[%(levelname)s::L%(lineno)d] %(message)s"

    FORMATS = {
        logging.INFO: blue + _format + reset,
        logging.DEBUG: yellow + _format + reset,
        logging.WARNING: magenta + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: red + _format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def flatten(arr: Iterable):
    for it in arr:
        if isinstance(it, Iterable) and not isinstance(it, str):
            yield from flatten(it)
        else:
            yield it


def mapfunc(func: Callable, obj: MutableMapping):
    for k, v in obj.items():
        if isinstance(v, MutableMapping):
            mapfunc(func, v)
        elif not isinstance(v, str):
            obj[k] = func(v)
        continue


def maxkeys(di: Dict) -> List:
    if any(isinstance(v, dict) for v in di.values()):
        for v in di.values():
            return maxkeys(v)
    return max(di.items(), key=lambda x: len(set(x[0])))[1]


def walk(path: Path):
    for p in path.iterdir():
        if p.is_dir():
            yield from walk(p)
            continue
        yield p.resolve()
