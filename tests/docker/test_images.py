import pytest
from bentoml.docker import ImageProvider
from bentoml.docker.images import get_suffix

tags: str = "bentoml/model-server:{}"


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        (
            ['debian', '3.7'],
            {'gpu': False, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.7-slim-runtime'),
        ),
        (
            ['slim', '3.7'],
            {'gpu': True, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.7-slim-cudnn'),
        ),
        (
            ['ubuntu', '3.8'],
            {'gpu': True, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-slim-cudnn'),
        ),
        (
            ['amazonlinux', '3.8'],
            {'gpu': False, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-ami2-runtime'),
        ),
        (
            ['centos7', '3.8'],
            {'gpu': True, 'bentoml_version': '0.14.0'},
            tags.format('0.14.0-python3.8-centos7-cudnn'),
        ),
        (
            ['centos8', '3.7'],
            {'gpu': False, 'bentoml_version': '0.13.0'},
            tags.format('devel-python3.7-centos8'),
        ),
        (
            ['alpine3.14', '3.7'],
            {'gpu': False, 'bentoml_version': '0.15.1'},
            tags.format('0.15.1-python3.7-alpine3.14-runtime'),
        ),
    ],
)
def test_valid_combos(args, kwargs, expected):
    assert ImageProvider(*args, **kwargs) == expected


@pytest.mark.parametrize(
    'args, kwargs, expected',
    [
        (['debian', '3.6'], {'gpu': False, 'bentoml_version': '0.14.0'}, RuntimeError),
        (['alpine', '3.8'], {'gpu': True, 'bentoml_version': '0.13.0'}, RuntimeError),
        (['centos7', '3.8'], {'gpu': True, 'bentoml_version': '1.0.'}, ValueError),
        (
            ['amazonlinux', '3.8'],
            {'gpu': False, 'bentoml_version': '0.13.0'},
            RuntimeError,
        ),
    ],
)
def test_invalid_combos(args, kwargs, expected):
    with pytest.raises(expected):
        ImageProvider(*args, **kwargs)


def test_get_suffix():
    assert get_suffix(True) == 'cudnn'
    assert get_suffix(False) == 'runtime'
