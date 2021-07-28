# CSV utils following https://tools.ietf.org/html/rfc4180
from typing import Iterable, Iterator, Union


def csv_splitlines(string: str) -> Iterator[str]:
    if '"' in string:

        def _iter_line(line: str) -> Iterator[str]:
            quoted = False
            last_cur = 0
            for i, c in enumerate(line):
                if c == '"':
                    quoted = not quoted
                if not quoted and string[i : i + 1] == "\n":
                    if i == 0 or string[i - 1] != "\r":
                        yield line[last_cur:i]
                        last_cur = i + 1
                    else:
                        yield line[last_cur : i - 1]
                        last_cur = i + 1
            yield line[last_cur:]

        return _iter_line(string)

    return iter(string.splitlines())


def csv_split(string: str, delimiter: str) -> Iterator[str]:
    if '"' in string:
        dlen = len(delimiter)

        def _iter_line(line: str) -> Iterator[str]:
            quoted = False
            last_cur = 0
            for i, c in enumerate(line):
                if c == '"':
                    quoted = not quoted
                if not quoted and string[i : i + dlen] == delimiter:
                    yield line[last_cur:i]
                    last_cur = i + dlen
            yield line[last_cur:]

        return _iter_line(string)
    else:
        return iter(string.split(delimiter))


def csv_row(tds: Iterable) -> str:
    return ",".join(csv_quote(td) for td in tds)


def csv_unquote(string: str) -> str:
    if '"' in string:
        string = string.strip()
        assert string[0] == '"' and string[-1] == '"'
        return string[1:-1].replace('""', '"')
    return string


def csv_quote(td: Union[int, str]) -> str:
    """
    >>> csv_quote(1)
    '1'
    >>> csv_quote('string')
    'string'
    >>> csv_quote('a,b"c')
    '"a,b""c"'
    >>> csv_quote(' ')
    '" "'
    """
    if td is None:
        td = ""
    elif not isinstance(td, str):
        td = str(td)
    if "\n" in td or '"' in td or "," in td or not td.strip():
        return td.replace('"', '""').join('""')
    return td
