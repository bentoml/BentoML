import typing as t

def parse_options_header(
    value: t.AnyStr,
) -> tuple[bytes, dict[t.ByteString, t.ByteString]]: ...
