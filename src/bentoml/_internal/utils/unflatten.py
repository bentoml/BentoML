# -*- coding: utf-8 -*-
""" Unflatten nested dict/array data
*** This is a modified version of the original unflatten.py from https://github.com/dairiki/unflatten, which is
published under the license ***

Copyright (C) 2018 Geoffrey T. Dairiki
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

3. The name of the author may not be used to endorse or promote
   products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR `AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""
from __future__ import annotations
from __future__ import absolute_import

import re
import sys
import typing as t
from operator import itemgetter

if sys.version_info[0] == 2:
    string_type = basestring  # noqa: F821
else:
    string_type = str


def unflatten(arg: dict[str, t.Any]) -> dict[str, t.Any]:
    """Unflatten nested dict/array data.

    This function takes a single argument which may either be a
    ``dict`` (or any object having a dict-like ``.items()`` or
    ``.iteritems()`` method) or a sequence of ``(key, value)`` pairs.
    The keys in the ``dict`` or sequence should must all be strings.

    Examples
    --------

    Nested ``dict``\\s::

    >>> unflatten({'foo.bar': 'val'})
    {'foo': {'bar': 'val'}}

    Nested ``list``::

    >>> unflatten({'foo[0]': 'val', 'foo[1]': 'bar'})
    {'foo': ['val', 'bar']}

    Nested ``list``\\s::

    >>> unflatten({'foo[0][0]': 'val'})
    {'foo': [['val']]}

    Lists of ``dict``\\s::

    >>> unflatten({'foo[0].bar': 'val',
    ...            'foo[1].baz': 'x'})
    {'foo': [{'bar': 'val'}, {'baz': 'x'}]}

    """
    if hasattr(arg, "items"):
        items = arg.items()
    else:
        items = arg

    data: dict[str, t.Any] = {}
    holders: list[t.Any] = []
    for flat_key, val in items:
        parsed_key = _parse_key(flat_key)
        obj = data
        for depth, (key, next_key) in enumerate(zip(parsed_key, parsed_key[1:]), 1):
            if isinstance(next_key, string_type):
                holder_type = _dict_holder
            else:
                holder_type = _list_holder

            if key not in obj:
                obj[key] = holder_type(_unparse_key(parsed_key[:depth]))
                holders.append((obj, key))
            elif not isinstance(obj[key], holder_type):
                raise ValueError(
                    "conflicting types %s and %s for key %r"
                    % (
                        _node_type(obj[key]),
                        holder_type.node_type,
                        _unparse_key(parsed_key[:depth]),
                    )
                )
            obj = obj[key]

        last_key = parsed_key[-1]
        if isinstance(obj.get(last_key), _holder):
            raise ValueError(
                "conflicting types %s and terminal for key %r"
                % (_node_type(obj[last_key]), flat_key)
            )
        obj[last_key] = val

    for obj, key in reversed(holders):
        obj[key] = obj[key].getvalue()

    return data


def _node_type(value: _holder) -> tuple[object] | t.Literal["terminal"]:
    if isinstance(value, _holder):
        return (value.node_type,)
    else:
        return "terminal"


class _holder(dict):
    node_type: type

    def __init__(self, flat_key: str):
        self.flat_key = flat_key
        self.data: dict[t.Any, t.Any] = {}

    def __contains__(self, key: t.Any):
        return key in self.data

    def __getitem__(self, key: t.Any):
        return self.data[key]

    def get(self, key: t.Any):
        return self.data.get(key)

    def __setitem__(self, key: t.Any, value: t.Any):
        self.data[key] = value


class _dict_holder(_holder):
    node_type = dict

    def getvalue(self):
        return self.data


class _list_holder(_holder):
    node_type = list

    def getvalue(self) -> list[t.Any]:
        items = sorted(self.data.items(), key=itemgetter(0))
        value: list[t.Any] = []
        for n, (key, val) in enumerate(items):
            if key != n:
                assert key > n
                missing_key = "%s[%d]" % (self.flat_key, n)
                raise ValueError("missing key %r" % missing_key)
            value.append(val)
        return value


_dot_or_indexes_re = re.compile(r"(\.?\"[^\"]*\")|(\[\d+\])|(\.\w*)|(^\w*)")


def _parse_key(flat_key: str):
    if not isinstance(flat_key, string_type):
        raise TypeError("keys must be strings")

    split_key = _dot_or_indexes_re.split(flat_key)
    parts: list[t.Any] = [""] if flat_key.startswith(".") else []

    for i in range(0, len(split_key), 5):
        sep = split_key[i]

        if sep != "":
            raise ValueError("invalid separator %r in key %r" % (sep, flat_key))

        if len(split_key) < i + 4:
            break

        if split_key[i + 1] is not None:
            # quoted first string
            string = split_key[i + 1]
            if len(string) == 2:
                string = ""
            elif i == 0:
                string = string[2:-1] if string.startswith(".") else string[1:-1]
            else:
                if string[0] != ".":
                    raise ValueError("invalid string %r in key %r" % (string, flat_key))
                string = string[2:-1]
            parts.append(string)

        elif split_key[i + 2] is not None:
            # index
            parts.append(int(split_key[i + 2][1:-1]))

        elif split_key[i + 3] is not None or split_key[i + 4] is not None:
            # unquoted string
            string = split_key[i + 3] or split_key[i + 4] or ""
            if len(string) == 0:
                string = ""
            elif i == 0:
                string = string[1:] if string.startswith(".") else string
            else:
                if string[0] != ".":
                    raise ValueError("invalid string %r in key %r" % (string, flat_key))
                string = string[1:]
            parts.append(string)
        else:
            assert False

    if len(parts) > 0 and isinstance(parts[0], int):
        parts.insert(0, "")
    return parts


def _unparse_key(parsed: list[t.Any]) -> str:
    bits: list[str] = []
    for part in parsed:
        if isinstance(part, string_type):
            if part.isidentifier():
                fmt = ".%s" if bits else "%s"
            elif part == "":
                fmt = ".%s" if bits else "%s"
            else:
                fmt = '."%s"' if bits else '"%s"'
        else:
            fmt = "[%d]"
        bits.append(fmt % part)
    return "".join(bits)
