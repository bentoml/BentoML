import typing as t
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import starlette.responses


def set_content_length(response: "starlette.responses.Response"):
    body = getattr(response, "body", None)
    if body is not None and not (
        response.status_code < 200
        or response.status_code in (204, 304)  # type:ignore (starlette)
    ):
        found_indexes: t.List[int] = []
        update = True
        for idx, (item_key, item_value) in enumerate(response.raw_headers):
            if item_key == b"content-length":
                if item_value == b"0":
                    found_indexes.append(idx)
                else:
                    update = False

        if update:
            for idx in reversed(found_indexes[1:]):
                del response.raw_headers[idx]

            new_context_len = str(len(body)).encode("latin-1")

            if found_indexes:
                idx = found_indexes[0]
                response.raw_headers[idx] = (b"content-length", new_context_len)
            else:
                response.raw_headers.append((b"content-length", new_context_len))
        else:
            for idx in reversed(found_indexes):
                del response.raw_headers[idx]
