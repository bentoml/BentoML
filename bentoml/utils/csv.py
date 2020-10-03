# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# CSV utils following https://tools.ietf.org/html/rfc4180
from typing import Iterable, Iterator


def csv_splitlines(string) -> Iterator[str]:
    if '"' in string:

        def _iter_line(line):
            quoted = False
            last_cur = 0
            for i, c in enumerate(line):
                if c == '"':
                    quoted = not quoted
                if not quoted and string[i : i + 1] == "\n":
                    if i == 0 or string[i - 1] != '\r':
                        yield line[last_cur:i]
                        last_cur = i + 1
                    else:
                        yield line[last_cur : i - 1]
                        last_cur = i + 1
            yield line[last_cur:]

        return _iter_line(string)

    return iter(string.splitlines())


def csv_split(string, delimiter) -> Iterator[str]:
    if '"' in string:
        dlen = len(delimiter)

        def _iter_line(line):
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


def csv_row(tds: Iterable):
    return ",".join(csv_quote(td) for td in tds)


def csv_unquote(string):
    if '"' in string:
        string = string.strip()
        assert string[0] == '"' and string[-1] == '"'
        return string[1:-1].replace('""', '"')
    return string


def csv_quote(td):
    '''
    >>> csv_quote(1)
    '1'
    >>> csv_quote('string')
    'string'
    >>> csv_quote('a,b"c')
    '"a,b""c"'
    >>> csv_quote(' ')
    '" "'
    '''
    if td is None:
        td = ''
    elif not isinstance(td, str):
        td = str(td)
    if '\n' in td or '"' in td or ',' in td or not td.strip():
        return td.replace('"', '""').join('""')
    return td
