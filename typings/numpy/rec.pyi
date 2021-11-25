from typing import List

__all__: List[str]

def fromarrays(
    arrayList,
    dtype=...,
    shape=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
): ...
def fromrecords(
    recList,
    dtype=...,
    shape=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
): ...
def fromstring(
    datastring,
    dtype=...,
    shape=...,
    offset=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
): ...
def fromfile(
    fd,
    dtype=...,
    shape=...,
    offset=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
): ...
def array(
    obj,
    dtype=...,
    shape=...,
    offset=...,
    strides=...,
    formats=...,
    names=...,
    titles=...,
    aligned=...,
    byteorder=...,
    copy=...,
): ...
