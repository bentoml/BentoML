from typing import Any

from .error import YAMLError

class ResolverError(YAMLError): ...

class BaseResolver:
    DEFAULT_SCALAR_TAG: Any
    DEFAULT_SEQUENCE_TAG: Any
    DEFAULT_MAPPING_TAG: Any
    yaml_implicit_resolvers: Any
    yaml_path_resolvers: Any
    resolver_exact_paths: Any
    resolver_prefix_paths: Any

    def __init__(self) -> None: ...

class Resolver(BaseResolver): ...
