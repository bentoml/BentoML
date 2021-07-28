from typing import Tuple, Optional


def parse_grpc_url(url: str) -> Tuple[Optional[str], str]:
    """
    >>> parse_grpc_url("grpcs://yatai.com:43/query")
    ('grpcs', 'yatai.com:43/query')
    >>> parse_grpc_url("yatai.com:43/query")
    (None, 'yatai.com:43/query')
    """
    from urllib3.util import parse_url

    parts = parse_url(url)
    return parts.scheme, url.replace(f"{parts.scheme}://", "", 1)
