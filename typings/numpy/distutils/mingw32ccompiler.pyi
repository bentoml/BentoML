import platform
import sys

"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""

def get_msvcr_replacement():
    """Replacement for outdated version of get_msvcr from cygwinccompiler"""
    ...

_START = ...
_TABLE = ...

class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """A modified MingW32 compiler compatible with an MSVC built Python."""

    compiler_type = ...
    def __init__(self, verbose=..., dry_run=..., force=...) -> None: ...
    def link(
        self,
        target_desc,
        objects,
        output_filename,
        output_dir,
        libraries,
        library_dirs,
        runtime_library_dirs,
        export_symbols=...,
        debug=...,
        extra_preargs=...,
        extra_postargs=...,
        build_temp=...,
        target_lang=...,
    ): ...
    def object_filenames(self, source_filenames, strip_dir=..., output_dir=...): ...

def find_python_dll(): ...
def dump_table(dll): ...
def generate_def(dll, dfile):
    """Given a dll file location,  get all its exported symbols and dump them
    into the given def file.

    The .def file will be overwritten"""
    ...

def find_dll(dll_name): ...
def build_msvcr_library(debug=...): ...
def build_import_library(): ...

_MSVCRVER_TO_FULLVER = ...
if sys.platform == "win32": ...

def msvc_manifest_xml(maj, min):
    """Given a major and minor version of the MSVCR, returns the
    corresponding XML file."""
    ...

def manifest_rc(name, type=...):
    """Return the rc file used to generate the res file which will be embedded
    as manifest for given manifest file name, of given type ('dll' or
    'exe').

    Parameters
    ----------
    name : str
            name of the manifest file to embed
    type : str {'dll', 'exe'}
            type of the binary which will embed the manifest

    """
    ...

def check_embedded_msvcr_match_linked(msver):
    """msver is the ms runtime version used for the MANIFEST."""
    ...

def configtest_name(config): ...
def manifest_name(config): ...
def rc_name(config): ...
def generate_manifest(config): ...
