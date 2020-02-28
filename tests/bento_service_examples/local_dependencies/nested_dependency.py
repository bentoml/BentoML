from tests.bento_service_examples.local_dependencies.secondary_dependency import (
    secondary_dependency_func,
)


def nested_dependency_func(foo):
    return secondary_dependency_func((foo))
