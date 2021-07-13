import pytest
from platform import python_version
from bentoml.docker import ImageProvider
from bentoml.docker.images import get_suffix, SUPPORTED_PYTHON_VERSION
from bentoml import __version__ as BENTOML_VERSION

tags: str = "bentoml/model-server:{}"
ver: str = BENTOML_VERSION.split("+")[0]
major, minor, _ = ver.split(".")
py_ver: str = python_version().split('.', maxsplit=1)[0]


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        (
            ['debian'],
            {'python_version': '3.6', 'gpu': False, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.6-slim-runtime'),
        ),
        (
            pytest.param(
                ['debian'],
                {'gpu': False, 'bentoml_version': '0.14.0'},
                tags.format(f'0.14.0-python{py_ver}-slim-runtime'),
                marks=pytest.mark.skipif(
                    py_ver not in SUPPORTED_PYTHON_VERSION,
                    reason='only test on supported python version',
                ),
            )
        ),
        (
            pytest.param(
                ['debian'],
                {'gpu': False, 'python_version': '3.6'},
                tags.format(f'{ver}-python3.6-slim-runtime'),
                marks=pytest.mark.skipif(
                    int(major) == 0 and int(minor) < 14,
                    reason='format in older than 0.14.0 will default to devel',
                ),
            )
        ),
        (
            ['slim'],
            {'gpu': True, 'python_version': '3.7', 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.7-slim-cudnn'),
        ),
        (
            ['ubuntu'],
            {'gpu': True, 'python_version': '3.8', 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-slim-cudnn'),
        ),
        (
            ['amazonlinux'],
            {'gpu': False, 'python_version': '3.8', 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-ami2-runtime'),
        ),
        (
            ['centos7'],
            {'gpu': True, 'python_version': '3.8', 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-centos7-cudnn'),
        ),
        (
            ['centos8'],
            {'gpu': False, 'python_version': '3.7', 'bentoml_version': '0.13.0'},
            tags.format('devel-python3.7-centos8'),
        ),
        (
            ['alpine3.14'],
            {'gpu': False, 'python_version': '3.7', 'bentoml_version': '0.15.1'},
            tags.format('0.15.1-python3.7-alpine3.14-runtime'),
        ),
    ],
)
def test_valid_combos(args, kwargs, expected):
    assert ImageProvider(*args, **kwargs) == expected


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        (
            ['debian'],
            {'gpu': False, 'python_version': '3.5', 'bentoml_version': '0.14.0'},
            RuntimeError,
        ),
        (
            ['alpine'],
            {'gpu': True, 'python_version': '3.8', 'bentoml_version': '0.13.0'},
            RuntimeError,
        ),
        (
            ['centos7'],
            {'gpu': True, 'python_version': '3.8', 'bentoml_version': '1.0.'},
            ValueError,
        ),
        (
            ['ubi8'],
            {'gpu': True, 'python_version': '3.9', 'bentoml_version': '1.0.0'},
            RuntimeError,
        ),
        (['amazonlinux'], {'gpu': False, 'python_version': '3.8'}, RuntimeError),
    ],
)
def test_invalid_combos(args, kwargs, expected):
    with pytest.raises(expected):
        ImageProvider(*args, **kwargs)


def test_get_suffix():
    assert get_suffix(True) == 'cudnn'
    assert get_suffix(False) == 'runtime'
