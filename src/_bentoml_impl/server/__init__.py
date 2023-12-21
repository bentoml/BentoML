"""
    _bentoml_impl.server
    ~~~~~~~~~~~~~~~~~~~~

    A reference implementation of serving a BentoML service.
    This will be eventually migrated to Rust.
"""
from .serving import serve_http

__all__ = ["serve_http"]
