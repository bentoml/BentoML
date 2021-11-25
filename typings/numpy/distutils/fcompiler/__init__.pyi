import os
import re
import sys
from distutils.errors import (
    CompileError,
    DistutilsExecError,
    DistutilsModuleError,
    DistutilsPlatformError,
    LinkError,
)
from distutils.fancy_getopt import FancyGetopt
from distutils.sysconfig import get_python_lib
from distutils.util import split_quoted, strtobool

from numpy.distutils import _shell_utils, log
from numpy.distutils.ccompiler import CCompiler, gen_lib_options
from numpy.distutils.exec_command import find_executable
from numpy.distutils.misc_util import (
    all_strings,
    get_shared_lib_extension,
    is_sequence,
    is_string,
    make_temp_file,
)

from .environment import EnvironmentConfig

"""numpy.distutils.fcompiler

Contains FCompiler, an abstract base class that defines the interface
for the numpy.distutils Fortran compiler abstraction model.

Terminology:

To be consistent, where the term 'executable' is used, it means the single
file, like 'gcc', that is executed, and should be a string. In contrast,
'command' means the entire command line, like ['gcc', '-c', 'file.c'], and
should be a list.

But note that FCompiler.executables is actually a dictionary of commands.

"""
__all__ = ["FCompiler", "new_fcompiler", "show_fcompilers", "dummy_fortran_file"]
__metaclass__ = type

class CompilerNotFound(Exception): ...

def flaglist(s): ...
def str2bool(s): ...
def is_sequence_of_strings(seq): ...

class FCompiler(CCompiler):
    """Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        update_executables(), find_executables(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except update_executables() and find_executables(),
    may call the get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """

    distutils_vars = ...
    command_vars = ...
    flag_vars = ...
    language_map = ...
    language_order = ...
    compiler_type = ...
    compiler_aliases = ...
    version_pattern = ...
    possible_executables = ...
    executables = ...
    suggested_f90_compiler = ...
    compile_switch = ...
    object_switch = ...
    library_switch = ...
    module_dir_switch = ...
    module_include_switch = ...
    pic_flags = ...
    src_extensions = ...
    obj_extension = ...
    shared_lib_extension = ...
    static_lib_extension = ...
    static_lib_format = ...
    shared_lib_format = ...
    exe_extension = ...
    _exe_cache = ...
    _executable_keys = ...
    c_compiler = ...
    extra_f77_compile_args = ...
    extra_f90_compile_args = ...
    def __init__(self, *args, **kw) -> None: ...
    def __copy__(self): ...
    def copy(self): ...
    version_cmd = ...
    compiler_f77 = ...
    compiler_f90 = ...
    compiler_fix = ...
    linker_so = ...
    linker_exe = ...
    archiver = ...
    ranlib = ...
    def set_executable(self, key, value): ...
    def set_commands(self, **kw): ...
    def set_command(self, key, value): ...
    def find_executables(self):
        """Go through the self.executables dictionary, and attempt to
        find and assign appropriate executables.

        Executable names are looked for in the environment (environment
        variables, the distutils.cfg, and command line), the 0th-element of
        the command list, and the self.possible_executables list.

        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
        or the Fortran 90 compiler executable is used, unless overridden
        by an environment setting.

        Subclasses should call this if overridden.
        """
        ...
    def update_executables(self):
        """Called at the beginning of customisation. Subclasses should
        override this if they need to set up the executables dictionary.

        Note that self.find_executables() is run afterwards, so the
        self.executables dictionary values can contain <F77> or <F90> as
        the command, which will be replaced by the found F77 or F90
        compiler.
        """
        ...
    def get_flags(self):
        """List of flags common to all compiler types."""
        ...
    def get_flags_f77(self):
        """List of Fortran 77 specific flags."""
        ...
    def get_flags_f90(self):
        """List of Fortran 90 specific flags."""
        ...
    def get_flags_free(self):
        """List of Fortran 90 free format specific flags."""
        ...
    def get_flags_fix(self):
        """List of Fortran 90 fixed format specific flags."""
        ...
    def get_flags_linker_so(self):
        """List of linker flags to build a shared library."""
        ...
    def get_flags_linker_exe(self):
        """List of linker flags to build an executable."""
        ...
    def get_flags_ar(self):
        """List of archiver flags."""
        ...
    def get_flags_opt(self):
        """List of architecture independent compiler flags."""
        ...
    def get_flags_arch(self):
        """List of architecture dependent compiler flags."""
        ...
    def get_flags_debug(self):
        """List of compiler flags to compile with debugging information."""
        ...
    get_flags_opt_f77 = ...
    get_flags_arch_f77 = ...
    get_flags_debug_f77 = ...
    def get_libraries(self):
        """List of compiler libraries."""
        ...
    def get_library_dirs(self):
        """List of compiler library directories."""
        ...
    def get_version(self, force=..., ok_status=...): ...
    def customize(self, dist=...):
        """Customize Fortran compiler.

        This method gets Fortran compiler specific information from
        (i) class definition, (ii) environment, (iii) distutils config
        files, and (iv) command line (later overrides earlier).

        This method should be always called after constructing a
        compiler instance. But not in __init__ because Distribution
        instance is needed for (iii) and (iv).
        """
        ...
    def dump_properties(self):
        """Print out the attributes of a compiler instance."""
        ...
    def module_options(self, module_dirs, module_build_dir): ...
    def library_option(self, lib): ...
    def library_dir_option(self, dir): ...
    def link(
        self,
        target_desc,
        objects,
        output_filename,
        output_dir=...,
        libraries=...,
        library_dirs=...,
        runtime_library_dirs=...,
        export_symbols=...,
        debug=...,
        extra_preargs=...,
        extra_postargs=...,
        build_temp=...,
        target_lang=...,
    ): ...
    def can_ccompiler_link(self, ccompiler):
        """
        Check if the given C compiler can link objects produced by
        this compiler.
        """
        ...
    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.

        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.

        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.

        """
        ...

_default_compilers = ...
fcompiler_class = ...
fcompiler_aliases = ...

def load_all_fcompiler_classes():
    """Cache all the FCompiler classes found in modules in the
    numpy.distutils.fcompiler package.
    """
    ...

def available_fcompilers_for_platform(osname=..., platform=...): ...
def get_default_fcompiler(osname=..., platform=..., requiref90=..., c_compiler=...):
    """Determine the default Fortran compiler to use for the given
    platform."""
    ...

failed_fcompilers = ...

def new_fcompiler(
    plat=...,
    compiler=...,
    verbose=...,
    dry_run=...,
    force=...,
    requiref90=...,
    c_compiler=...,
):
    """Generate an instance of some FCompiler subclass for the supplied
    platform/compiler combination.
    """
    ...

def show_fcompilers(dist=...):
    """Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """
    ...

def dummy_fortran_file(): ...

is_f_file = ...
_has_f_header = ...
_has_f90_header = ...
_has_fix_header = ...
_free_f90_start = ...

def is_free_format(file):
    """Check if file is in free format Fortran."""
    ...

def has_f90_header(src): ...

_f77flags_re = ...

def get_f77flags(src):
    """
    Search the first 20 lines of fortran 77 code for line pattern
      `CF77FLAGS(<fcompiler type>)=<f77 flags>`
    Return a dictionary {<fcompiler type>:<f77 flags>}.
    """
    ...

if __name__ == "__main__": ...
