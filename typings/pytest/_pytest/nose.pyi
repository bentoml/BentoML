
from _pytest.config import hookimpl
from _pytest.nodes import Item

"""Run testsuites written for nose."""
@hookimpl(trylast=True)
def pytest_runtest_setup(item): # -> None:
    ...

def teardown_nose(item): # -> None:
    ...

def is_potential_nosetest(item: Item) -> bool:
    ...

def call_optional(obj, name): # -> Literal[True] | None:
    ...

