# CSV utils following https://tools.ietf.org/html/rfc4180
import typing as t


def csv_splitlines(string: str) -> t.Iterator[str]:
    if '"' in string:

        def _iter_line(line: str) -> t.Iterator[str]:
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


def csv_split(string: str, delimiter: str) -> t.Iterator[str]:
    if '"' in string:
        d_len = len(delimiter)

        def _iter_line(line: str) -> t.Iterator[str]:
            quoted = False
            last_cur = 0
            for i, c in enumerate(line):
                if c == '"':
                    quoted = not quoted
                if not quoted and string[i : i + d_len] == delimiter:
                    yield line[last_cur:i]
                    last_cur = i + d_len
            yield line[last_cur:]

        return _iter_line(string)
    else:
        return iter(string.split(delimiter))


def csv_row(tds: t.Iterable) -> str:
    return ",".join(csv_quote(td) for td in tds)


def csv_unquote(string: str) -> str:
    if '"' in string:
        string = string.strip()
        assert string[0] == '"' and string[-1] == '"'
        return string[1:-1].replace('""', '"')
    return string


def csv_quote(td: t.Union[int, str]) -> str:
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
