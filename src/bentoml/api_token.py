"""
User facing python APIs for API token management
"""

from __future__ import annotations

import typing as t
from datetime import datetime

from simple_di import Provide
from simple_di import inject

from ._internal.cloud.api_token import ApiToken
from ._internal.configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from ._internal.cloud import BentoCloudClient


@inject
def list(
    search: str | None = None,
    _cloud_client: "BentoCloudClient" = Provide[BentoMLContainer.bentocloud_client],
) -> t.List[ApiToken]:
    """List all API tokens.

    Args:
        search: Optional search string to filter tokens by name

    Returns:
        List of ApiToken objects

    Example:
        >>> import bentoml
        >>> tokens = bentoml.api_token.list(search="my-token")
        >>> for token in tokens:
        ...     print(f"{token.name}: {token.uid}")
    """
    return _cloud_client.api_token.list(search=search)


@inject
def create(
    name: str,
    description: str | None = None,
    scopes: t.List[str] | None = None,
    expired_at: datetime | None = None,
    _cloud_client: "BentoCloudClient" = Provide[BentoMLContainer.bentocloud_client],
) -> ApiToken:
    """Create a new API token.

    Args:
        name: Name of the token
        description: Optional description
        scopes: List of scopes. Available scopes:
            - api: General API access
            - read_organization: Read organization data
            - write_organization: Write organization data
            - read_cluster: Read cluster data
            - write_cluster: Write cluster data
        expired_at: Optional expiration datetime

    Returns:
        ApiToken object (includes token value - save it, it won't be shown again!)

    Example:
        >>> import bentoml
        >>> from datetime import datetime, timedelta
        >>> token = bentoml.api_token.create(
        ...     name="ci-token",
        ...     description="CI/CD pipeline token",
        ...     scopes=["api", "read_cluster"],
        ...     expired_at=datetime.now() + timedelta(days=30)
        ... )
        >>> print(f"Token: {token.token}")  # Save this!
    """
    return _cloud_client.api_token.create(
        name=name,
        description=description,
        scopes=scopes,
        expired_at=expired_at,
    )


@inject
def get(
    token_uid: str,
    _cloud_client: "BentoCloudClient" = Provide[BentoMLContainer.bentocloud_client],
) -> ApiToken | None:
    """Get an API token by UID.

    Args:
        token_uid: The UID of the token

    Returns:
        ApiToken object or None if not found

    Example:
        >>> import bentoml
        >>> token = bentoml.api_token.get("token_abc123")
        >>> if token:
        ...     print(f"Scopes: {token.scopes}")
    """
    return _cloud_client.api_token.get(token_uid=token_uid)


@inject
def delete(
    token_uid: str,
    _cloud_client: "BentoCloudClient" = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """Delete an API token.

    Args:
        token_uid: The UID of the token to delete

    Example:
        >>> import bentoml
        >>> bentoml.api_token.delete("token_abc123")
    """
    _cloud_client.api_token.delete(token_uid=token_uid)


__all__ = ["create", "get", "list", "delete"]
