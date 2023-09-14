from __future__ import annotations

from typing import TYPE_CHECKING

import attr

if TYPE_CHECKING:
    import starlette.responses


@attr.define
class Cookie:
    key: str
    value: str
    max_age: int | None
    expires: int | None
    path: str
    domain: str | None
    secure: bool
    httponly: bool
    samesite: str


def set_cookies(response: starlette.responses.Response, cookies: list[Cookie]):
    for cookie in cookies:
        response.set_cookie(
            cookie.key,
            cookie.value,
            max_age=cookie.max_age,  # type: ignore (bad starlette types)
            expires=cookie.expires,  # type: ignore (bad starlette types)
            path=cookie.path,
            domain=cookie.domain,  # type: ignore (bad starlette types)
            secure=cookie.secure,
            httponly=cookie.httponly,
            samesite=cookie.samesite,
        )
