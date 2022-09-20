# End-to-end tests suite

This folder contains end-to-end test suite.

## Instruction

To create a new test suite (for simplicity let's call our test suite `qa`), do the following:

1. Navigate to [`config.yml`](../../scripts/ci/config.yml) and add the E2E definition:

```yaml
qa:
  <<: *tmpl
  root_test_dir: "tests/e2e/qa"
  is_dir: true
  type_tests: "e2e"
  dependencies:  # add required Python dependencies here.
    - Pillow
    - pydantic
    - grpcio-status
```

2. Create the folder `qa` with the following project structure:

```bash
.
├── bentofile.yaml
├── train.py
...
├── service.py
└── tests
    ├── conftest.py
    ├── test_io.py
    ...
    └── test_meta.py
```

> Note that files under `tests` are merely examples, feel free to add any types of
> additional tests.

3. Create a `train.py`:

```python
if __name__ == "__main__":
    import python_model

    import bentoml

    bentoml.picklable_model.save_model(
        "py_model.case-1.grpc.e2e",
        python_model.PythonFunction(),
        signatures={
            "echo_json": {"batchable": True},
            "echo_object": {"batchable": False},
            "echo_ndarray": {"batchable": True},
            "double_ndarray": {"batchable": True},
        },
        external_modules=[python_model],
    )
```

4. Inside `tests/conftest.py`, create a `host` fixture like so:

```python
# pylint: disable=unused-argument
from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

import pytest

from bentoml._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from contextlib import ExitStack

    from _pytest.main import Session
    from _pytest.nodes import Item
    from _pytest.config import Config


def pytest_collection_modifyitems(
    session: Session, config: Config, items: list[Item]
) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "train"],
        env={"BENTOML_HOME": BentoMLContainer.bentoml_home.get()},
    )


@pytest.fixture(scope="module")
def host(
    bentoml_home: str,
    deployment_mode: str,
    clean_context: ExitStack,
) -> t.Generator[str, None, None]:
    from bentoml.testing.server import host_bento

    with host_bento(
        "service:svc",
        deployment_mode=deployment_mode,
        bentoml_home=bentoml_home,
        clean_context=clean_context,
        use_grpc=True,
    ) as _host:
        yield _host
```

5. To run the tests, navigate to `GIT_ROOT` (root directory of bentoml), and call:

```bash
./scripts/ci/run_tests.sh qa
```

By default, the E2E suite is setup so that the models and bentos will be created and
saved under pytest temporary directory. To cleanup after the test, passing `--cleanup`
to `run_tests.sh`:

```bash
./scripts/ci/run_tests.sh qa --cleanup
```
