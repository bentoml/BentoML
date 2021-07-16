from tests.bento_services.local_dependencies.secondary_dependency import (
    secondary_dependency_func,
)


def nested_dependency_func(foo):
    return secondary_dependency_func((foo))
