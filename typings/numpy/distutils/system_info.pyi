import os
import platform
import sys
from distutils.errors import DistutilsError

"""
This file defines a set of system_info classes for getting
information about various resources (libraries, library directories,
include directories, etc.) in the system. Usage:
    info_dict = get_info(<name>)
  where <name> is a string 'atlas','x11','fftw','lapack','blas',
  'lapack_src', 'blas_src', etc. For a complete list of allowed names,
  see the definition of get_info() function below.

  Returned info_dict is a dictionary which is compatible with
  distutils.setup keyword arguments. If info_dict == {}, then the
  asked resource is not available (system_info could not find it).

  Several *_info classes specify an environment variable to specify
  the locations of software. When setting the corresponding environment
  variable to 'None' then the software will be ignored, even when it
  is available in system.

Global parameters:
  system_info.search_static_first - search static libraries (.a)
             in precedence to shared ones (.so, .sl) if enabled.
  system_info.verbosity - output the results to stdout if enabled.

The file 'site.cfg' is looked for in

1) Directory of main setup.py file being run.
2) Home directory of user running the setup.py file as ~/.numpy-site.cfg
3) System wide directory (location of this file...)

The first one found is used to get system configuration options The
format is that used by ConfigParser (i.e., Windows .INI style). The
section ALL is not intended for general use.

Appropriate defaults are used if nothing is specified.

The order of finding the locations of resources is the following:
 1. environment variable
 2. section in site.cfg
 3. DEFAULT section in site.cfg
 4. System default search paths (see ``default_*`` variables below).
Only the first complete match is returned.

Currently, the following classes are available, along with their section names:

    Numeric_info:Numeric
    _numpy_info:Numeric
    _pkg_config_info:None
    accelerate_info:accelerate
    agg2_info:agg2
    amd_info:amd
    atlas_3_10_blas_info:atlas
    atlas_3_10_blas_threads_info:atlas
    atlas_3_10_info:atlas
    atlas_3_10_threads_info:atlas
    atlas_blas_info:atlas
    atlas_blas_threads_info:atlas
    atlas_info:atlas
    atlas_threads_info:atlas
    blas64__opt_info:ALL               # usage recommended (general ILP64 BLAS, 64_ symbol suffix)
    blas_ilp64_opt_info:ALL            # usage recommended (general ILP64 BLAS)
    blas_ilp64_plain_opt_info:ALL      # usage recommended (general ILP64 BLAS, no symbol suffix)
    blas_info:blas
    blas_mkl_info:mkl
    blas_opt_info:ALL                  # usage recommended
    blas_src_info:blas_src
    blis_info:blis
    boost_python_info:boost_python
    dfftw_info:fftw
    dfftw_threads_info:fftw
    djbfft_info:djbfft
    f2py_info:ALL
    fft_opt_info:ALL
    fftw2_info:fftw
    fftw3_info:fftw3
    fftw_info:fftw
    fftw_threads_info:fftw
    flame_info:flame
    freetype2_info:freetype2
    gdk_2_info:gdk_2
    gdk_info:gdk
    gdk_pixbuf_2_info:gdk_pixbuf_2
    gdk_pixbuf_xlib_2_info:gdk_pixbuf_xlib_2
    gdk_x11_2_info:gdk_x11_2
    gtkp_2_info:gtkp_2
    gtkp_x11_2_info:gtkp_x11_2
    lapack64__opt_info:ALL             # usage recommended (general ILP64 LAPACK, 64_ symbol suffix)
    lapack_atlas_3_10_info:atlas
    lapack_atlas_3_10_threads_info:atlas
    lapack_atlas_info:atlas
    lapack_atlas_threads_info:atlas
    lapack_ilp64_opt_info:ALL          # usage recommended (general ILP64 LAPACK)
    lapack_ilp64_plain_opt_info:ALL    # usage recommended (general ILP64 LAPACK, no symbol suffix)
    lapack_info:lapack
    lapack_mkl_info:mkl
    lapack_opt_info:ALL                # usage recommended
    lapack_src_info:lapack_src
    mkl_info:mkl
    numarray_info:numarray
    numerix_info:numerix
    numpy_info:numpy
    openblas64__info:openblas64_
    openblas64__lapack_info:openblas64_
    openblas_clapack_info:openblas
    openblas_ilp64_info:openblas_ilp64
    openblas_ilp64_lapack_info:openblas_ilp64
    openblas_info:openblas
    openblas_lapack_info:openblas
    sfftw_info:fftw
    sfftw_threads_info:fftw
    system_info:ALL
    umfpack_info:umfpack
    wx_info:wx
    x11_info:x11
    xft_info:xft

Note that blas_opt_info and lapack_opt_info honor the NPY_BLAS_ORDER
and NPY_LAPACK_ORDER environment variables to determine the order in which
specific BLAS and LAPACK libraries are searched for.

This search (or autodetection) can be bypassed by defining the environment
variables NPY_BLAS_LIBS and NPY_LAPACK_LIBS, which should then contain the
exact linker flags to use (language will be set to F77). Building against
Netlib BLAS/LAPACK or stub files, in order to be able to switch BLAS and LAPACK
implementations at runtime. If using this to build NumPy itself, it is
recommended to also define NPY_CBLAS_LIBS (assuming your BLAS library has a
CBLAS interface) to enable CBLAS usage for matrix multiplication (unoptimized
otherwise).

Example:
----------
[DEFAULT]
# default section
library_dirs = /usr/lib:/usr/local/lib:/opt/lib
include_dirs = /usr/include:/usr/local/include:/opt/include
src_dirs = /usr/local/src:/opt/src
# search static libraries (.a) in preference to shared ones (.so)
search_static_first = 0

[fftw]
libraries = rfftw, fftw

[atlas]
library_dirs = /usr/lib/3dnow:/usr/lib/3dnow/atlas
# for overriding the names of the atlas libraries
libraries = lapack, f77blas, cblas, atlas

[x11]
library_dirs = /usr/X11R6/lib
include_dirs = /usr/X11R6/include
----------

Note that the ``libraries`` key is the default setting for libraries.

Authors:
  Pearu Peterson <pearu@cens.ioc.ee>, February 2002
  David M. Cooke <cookedm@physics.mcmaster.ca>, April 2002

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

"""
__all__ = ["system_info"]
_bits = ...
platform_bits = ...
global_compiler = ...

def customized_ccompiler(): ...
def libpaths(paths, bits):
    """Return a list of library paths valid on 32 or 64 bit systems.

    Inputs:
      paths : sequence
        A sequence of strings (typically paths)
      bits : int
        An integer, the only valid values are 32 or 64.  A ValueError exception
      is raised otherwise.

    Examples:

    Consider a list of directories
    >>> paths = ['/usr/X11R6/lib','/usr/X11/lib','/usr/lib']

    For a 32-bit platform, this is already valid:
    >>> np.distutils.system_info.libpaths(paths,32)
    ['/usr/X11R6/lib', '/usr/X11/lib', '/usr/lib']

    On 64 bits, we prepend the '64' postfix
    >>> np.distutils.system_info.libpaths(paths,64)
    ['/usr/X11R6/lib64', '/usr/X11R6/lib', '/usr/X11/lib64', '/usr/X11/lib',
    '/usr/lib64', '/usr/lib']
    """
    ...

if sys.platform == "win32": ...
else:
    default_lib_dirs = ...
    default_runtime_dirs = ...
    default_include_dirs = ...
    default_src_dirs = ...
    default_x11_lib_dirs = ...
    default_x11_include_dirs = ...
if os.path.join(sys.prefix, "lib") not in default_lib_dirs: ...
default_lib_dirs = ...
default_runtime_dirs = ...
default_include_dirs = ...
default_src_dirs = ...
so_ext = ...

def get_standard_file(fname):
    """Returns a list of files named 'fname' from
    1) System-wide directory (directory-location of this module)
    2) Users HOME directory (os.environ['HOME'])
    3) Local directory
    """
    ...

def get_info(name, notfound_action=...):
    """
    notfound_action:
      0 - do nothing
      1 - display warning message
      2 - raise error
    """
    ...

class NotFoundError(DistutilsError):
    """Some third-party program or library is not found."""

    ...

class AliasedOptionError(DistutilsError):
    """
    Aliases entries in config files should not be existing.
    In section '{section}' we found multiple appearances of options {options}."""

    ...

class AtlasNotFoundError(NotFoundError):
    """
    Atlas (http://github.com/math-atlas/math-atlas) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [atlas]) or by setting
    the ATLAS environment variable."""

    ...

class FlameNotFoundError(NotFoundError):
    """
    FLAME (http://www.cs.utexas.edu/~flame/web/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [flame])."""

    ...

class LapackNotFoundError(NotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [lapack]) or by setting
    the LAPACK environment variable."""

    ...

class LapackSrcNotFoundError(LapackNotFoundError):
    """
    Lapack (http://www.netlib.org/lapack/) sources not found.
    Directories to search for the sources can be specified in the
    numpy/distutils/site.cfg file (section [lapack_src]) or by setting
    the LAPACK_SRC environment variable."""

    ...

class LapackILP64NotFoundError(NotFoundError):
    """
    64-bit Lapack libraries not found.
    Known libraries in numpy/distutils/site.cfg file are:
    openblas64_, openblas_ilp64
    """

    ...

class BlasOptNotFoundError(NotFoundError):
    """
    Optimized (vendor) Blas libraries are not found.
    Falls back to netlib Blas library which has worse performance.
    A better performance should be easily gained by switching
    Blas library."""

    ...

class BlasNotFoundError(NotFoundError):
    """
    Blas (http://www.netlib.org/blas/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [blas]) or by setting
    the BLAS environment variable."""

    ...

class BlasILP64NotFoundError(NotFoundError):
    """
    64-bit Blas libraries not found.
    Known libraries in numpy/distutils/site.cfg file are:
    openblas64_, openblas_ilp64
    """

    ...

class BlasSrcNotFoundError(BlasNotFoundError):
    """
    Blas (http://www.netlib.org/blas/) sources not found.
    Directories to search for the sources can be specified in the
    numpy/distutils/site.cfg file (section [blas_src]) or by setting
    the BLAS_SRC environment variable."""

    ...

class FFTWNotFoundError(NotFoundError):
    """
    FFTW (http://www.fftw.org/) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [fftw]) or by setting
    the FFTW environment variable."""

    ...

class DJBFFTNotFoundError(NotFoundError):
    """
    DJBFFT (https://cr.yp.to/djbfft.html) libraries not found.
    Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [djbfft]) or by setting
    the DJBFFT environment variable."""

    ...

class NumericNotFoundError(NotFoundError):
    """
    Numeric (https://www.numpy.org/) module not found.
    Get it from above location, install it, and retry setup.py."""

    ...

class X11NotFoundError(NotFoundError):
    """X11 libraries not found."""

    ...

class UmfpackNotFoundError(NotFoundError):
    """
    UMFPACK sparse solver (https://www.cise.ufl.edu/research/sparse/umfpack/)
    not found. Directories to search for the libraries can be specified in the
    numpy/distutils/site.cfg file (section [umfpack]) or by setting
    the UMFPACK environment variable."""

    ...

class system_info:
    """get_info() is the only public method. Don't use others."""

    dir_env_var = ...
    search_static_first = ...
    section = ...
    saved_results = ...
    notfounderror = NotFoundError
    def __init__(self, default_lib_dirs=..., default_include_dirs=...) -> None: ...
    def parse_config_files(self): ...
    def calc_libraries_info(self): ...
    def set_info(self, **info): ...
    def get_option_single(self, *options):
        """Ensure that only one of `options` are found in the section

        Parameters
        ----------
        *options : list of str
           a list of options to be found in the section (``self.section``)

        Returns
        -------
        str :
            the option that is uniquely found in the section

        Raises
        ------
        AliasedOptionError :
            in case more than one of the options are found
        """
        ...
    def has_info(self): ...
    def calc_extra_info(self):
        """Updates the information in the current information with
        respect to these flags:
          extra_compile_args
          extra_link_args
        """
        ...
    def get_info(self, notfound_action=...):
        """Return a dictionary with items that are compatible
        with numpy.distutils.setup keyword arguments.
        """
        ...
    def get_paths(self, section, key): ...
    def get_lib_dirs(self, key=...): ...
    def get_runtime_lib_dirs(self, key=...): ...
    def get_include_dirs(self, key=...): ...
    def get_src_dirs(self, key=...): ...
    def get_libs(self, key, default): ...
    def get_libraries(self, key=...): ...
    def library_extensions(self): ...
    def check_libs(self, lib_dirs, libs, opt_libs=...):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks for all libraries as shared libraries first, then
        static (or vice versa if self.search_static_first is True).
        """
        ...
    def check_libs2(self, lib_dirs, libs, opt_libs=...):
        """If static or shared libraries are available then return
        their info dictionary.

        Checks each library for shared or static.
        """
        ...
    def combine_paths(self, *args):
        """Return a list of existing paths composed by all combinations
        of items from the arguments.
        """
        ...

class fft_opt_info(system_info):
    def calc_info(self): ...

class fftw_info(system_info):
    section = ...
    dir_env_var = ...
    notfounderror = FFTWNotFoundError
    ver_info = ...
    def calc_ver_info(self, ver_param):
        """Returns True on successful version detection, else False"""
        ...
    def calc_info(self): ...

class fftw2_info(fftw_info):
    section = ...
    dir_env_var = ...
    notfounderror = FFTWNotFoundError
    ver_info = ...

class fftw3_info(fftw_info):
    section = ...
    dir_env_var = ...
    notfounderror = FFTWNotFoundError
    ver_info = ...

class dfftw_info(fftw_info):
    section = ...
    dir_env_var = ...
    ver_info = ...

class sfftw_info(fftw_info):
    section = ...
    dir_env_var = ...
    ver_info = ...

class fftw_threads_info(fftw_info):
    section = ...
    dir_env_var = ...
    ver_info = ...

class dfftw_threads_info(fftw_info):
    section = ...
    dir_env_var = ...
    ver_info = ...

class sfftw_threads_info(fftw_info):
    section = ...
    dir_env_var = ...
    ver_info = ...

class djbfft_info(system_info):
    section = ...
    dir_env_var = ...
    notfounderror = DJBFFTNotFoundError
    def get_paths(self, section, key): ...
    def calc_info(self): ...

class mkl_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_mkl = ...
    def get_mkl_rootdir(self): ...
    def __init__(self) -> None: ...
    def calc_info(self): ...

class lapack_mkl_info(mkl_info): ...
class blas_mkl_info(mkl_info): ...

class atlas_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    if sys.platform[:7] == "freebsd":
        _lib_atlas = ...
        _lib_lapack = ...
    else:
        _lib_atlas = ...
        _lib_lapack = ...
    notfounderror = AtlasNotFoundError
    def get_paths(self, section, key): ...
    def calc_info(self): ...

class atlas_blas_info(atlas_info):
    _lib_names = ...
    def calc_info(self): ...

class atlas_threads_info(atlas_info):
    dir_env_var = ...
    _lib_names = ...

class atlas_blas_threads_info(atlas_blas_info):
    dir_env_var = ...
    _lib_names = ...

class lapack_atlas_info(atlas_info):
    _lib_names = ...

class lapack_atlas_threads_info(atlas_threads_info):
    _lib_names = ...

class atlas_3_10_info(atlas_info):
    _lib_names = ...
    _lib_atlas = ...
    _lib_lapack = ...

class atlas_3_10_blas_info(atlas_3_10_info):
    _lib_names = ...
    def calc_info(self): ...

class atlas_3_10_threads_info(atlas_3_10_info):
    dir_env_var = ...
    _lib_names = ...
    _lib_atlas = ...
    _lib_lapack = ...

class atlas_3_10_blas_threads_info(atlas_3_10_blas_info):
    dir_env_var = ...
    _lib_names = ...

class lapack_atlas_3_10_info(atlas_3_10_info): ...
class lapack_atlas_3_10_threads_info(atlas_3_10_threads_info): ...

class lapack_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    notfounderror = LapackNotFoundError
    def calc_info(self): ...

class lapack_src_info(system_info):
    section = ...
    dir_env_var = ...
    notfounderror = LapackSrcNotFoundError
    def get_paths(self, section, key): ...
    def calc_info(self): ...

atlas_version_c_text = ...
_cached_atlas_version = ...

def get_atlas_version(**config): ...

class lapack_opt_info(system_info):
    notfounderror = LapackNotFoundError
    lapack_order = ...
    order_env_var_name = ...
    def calc_info(self): ...

class _ilp64_opt_info_mixin:
    symbol_suffix = ...
    symbol_prefix = ...

class lapack_ilp64_opt_info(lapack_opt_info, _ilp64_opt_info_mixin):
    notfounderror = LapackILP64NotFoundError
    lapack_order = ...
    order_env_var_name = ...

class lapack_ilp64_plain_opt_info(lapack_ilp64_opt_info):
    symbol_prefix = ...
    symbol_suffix = ...

class lapack64__opt_info(lapack_ilp64_opt_info):
    symbol_prefix = ...
    symbol_suffix = ...

class blas_opt_info(system_info):
    notfounderror = BlasNotFoundError
    blas_order = ...
    order_env_var_name = ...
    def calc_info(self): ...

class blas_ilp64_opt_info(blas_opt_info, _ilp64_opt_info_mixin):
    notfounderror = BlasILP64NotFoundError
    blas_order = ...
    order_env_var_name = ...

class blas_ilp64_plain_opt_info(blas_ilp64_opt_info):
    symbol_prefix = ...
    symbol_suffix = ...

class blas64__opt_info(blas_ilp64_opt_info):
    symbol_prefix = ...
    symbol_suffix = ...

class cblas_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    notfounderror = BlasNotFoundError

class blas_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    notfounderror = BlasNotFoundError
    def calc_info(self): ...
    def get_cblas_libs(self, info):
        """Check whether we can link with CBLAS interface

        This method will search through several combinations of libraries
        to check whether CBLAS is present:

        1. Libraries in ``info['libraries']``, as is
        2. As 1. but also explicitly adding ``'cblas'`` as a library
        3. As 1. but also explicitly adding ``'blas'`` as a library
        4. Check only library ``'cblas'``
        5. Check only library ``'blas'``

        Parameters
        ----------
        info : dict
           system information dictionary for compilation and linking

        Returns
        -------
        libraries : list of str or None
            a list of libraries that enables the use of CBLAS interface.
            Returns None if not found or a compilation error occurs.

            Since 1.17 returns a list.
        """
        ...

class openblas_info(blas_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    _require_symbols = ...
    notfounderror = BlasNotFoundError
    @property
    def symbol_prefix(self): ...
    @property
    def symbol_suffix(self): ...
    def calc_info(self): ...
    def check_msvc_gfortran_libs(self, library_dirs, libraries): ...
    def check_symbols(self, info): ...

class openblas_lapack_info(openblas_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    _require_symbols = ...
    notfounderror = BlasNotFoundError

class openblas_clapack_info(openblas_lapack_info):
    _lib_names = ...

class openblas_ilp64_info(openblas_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    _require_symbols = ...
    notfounderror = BlasILP64NotFoundError

class openblas_ilp64_lapack_info(openblas_ilp64_info):
    _require_symbols = ...

class openblas64__info(openblas_ilp64_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    symbol_suffix = ...
    symbol_prefix = ...

class openblas64__lapack_info(openblas_ilp64_lapack_info, openblas64__info): ...

class blis_info(blas_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    notfounderror = BlasNotFoundError
    def calc_info(self): ...

class flame_info(system_info):
    """Usage of libflame for LAPACK operations

    This requires libflame to be compiled with lapack wrappers:

    ./configure --enable-lapack2flame ...

    Be aware that libflame 5.1.0 has some missing names in the shared library, so
    if you have problems, try the static flame library.
    """

    section = ...
    _lib_names = ...
    notfounderror = FlameNotFoundError
    def check_embedded_lapack(self, info):
        """libflame does not necessarily have a wrapper for fortran LAPACK, we need to check"""
        ...
    def calc_info(self): ...

class accelerate_info(system_info):
    section = ...
    _lib_names = ...
    notfounderror = BlasNotFoundError
    def calc_info(self): ...

class blas_src_info(system_info):
    section = ...
    dir_env_var = ...
    notfounderror = BlasSrcNotFoundError
    def get_paths(self, section, key): ...
    def calc_info(self): ...

class x11_info(system_info):
    section = ...
    notfounderror = X11NotFoundError
    _lib_names = ...
    def __init__(self) -> None: ...
    def calc_info(self): ...

class _numpy_info(system_info):
    section = ...
    modulename = ...
    notfounderror = NumericNotFoundError
    def __init__(self) -> None: ...
    def calc_info(self): ...

class numarray_info(_numpy_info):
    section = ...
    modulename = ...

class Numeric_info(_numpy_info):
    section = ...
    modulename = ...

class numpy_info(_numpy_info):
    section = ...
    modulename = ...

class numerix_info(system_info):
    section = ...
    def calc_info(self): ...

class f2py_info(system_info):
    def calc_info(self): ...

class boost_python_info(system_info):
    section = ...
    dir_env_var = ...
    def get_paths(self, section, key): ...
    def calc_info(self): ...

class agg2_info(system_info):
    section = ...
    dir_env_var = ...
    def get_paths(self, section, key): ...
    def calc_info(self): ...

class _pkg_config_info(system_info):
    section = ...
    config_env_var = ...
    default_config_exe = ...
    append_config_exe = ...
    version_macro_name = ...
    release_macro_name = ...
    version_flag = ...
    cflags_flag = ...
    def get_config_exe(self): ...
    def get_config_output(self, config_exe, option): ...
    def calc_info(self): ...

class wx_info(_pkg_config_info):
    section = ...
    config_env_var = ...
    default_config_exe = ...
    append_config_exe = ...
    version_macro_name = ...
    release_macro_name = ...
    version_flag = ...
    cflags_flag = ...

class gdk_pixbuf_xlib_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gdk_pixbuf_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gdk_x11_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gdk_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gdk_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gtkp_x11_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class gtkp_2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class xft_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class freetype2_info(_pkg_config_info):
    section = ...
    append_config_exe = ...
    version_macro_name = ...

class amd_info(system_info):
    section = ...
    dir_env_var = ...
    _lib_names = ...
    def calc_info(self): ...

class umfpack_info(system_info):
    section = ...
    dir_env_var = ...
    notfounderror = UmfpackNotFoundError
    _lib_names = ...
    def calc_info(self): ...

def combine_paths(*args, **kws):
    """Return a list of existing paths composed by all combinations of
    items from arguments.
    """
    ...

language_map = ...
inv_language_map = ...

def dict_append(d, **kws): ...
def parseCmdLine(argv=...): ...
def show_all(argv=...): ...

if __name__ == "__main__": ...
