import sys

__all__ = ["get_info", "show"]
extra_dll_dir = ...
if sys.platform == "win32" and os.path.isdir(extra_dll_dir): ...
blas_mkl_info = ...
blis_info = ...
openblas_info = ...
blas_opt_info = ...
lapack_mkl_info = ...
openblas_lapack_info = ...
lapack_opt_info = ...

def get_info(name): ...
def show():  # -> None:
    """
    Show libraries in the system on which NumPy was built.

    Print information about various resources (libraries, library
    directories, include directories, etc.) in the system on which
    NumPy was built.

    See Also
    --------
    get_include : Returns the directory containing NumPy C
                  header files.

    Notes
    -----
    Classes specifying the information to be printed are defined
    in the `numpy.distutils.system_info` module.

    Information may include:

    * ``language``: language used to write the libraries (mostly
      C or f77)
    * ``libraries``: names of libraries found in the system
    * ``library_dirs``: directories containing the libraries
    * ``include_dirs``: directories containing library header files
    * ``src_dirs``: directories containing library source files
    * ``define_macros``: preprocessor macros used by
      ``distutils.setup``
    * ``baseline``: minimum CPU features required
    * ``found``: dispatched features supported in the system
    * ``not found``: dispatched features that are not supported
      in the system

    Examples
    --------
    >>> import numpy as np
    >>> np.show_config()
    blas_opt_info:
        language = c
        define_macros = [('HAVE_CBLAS', None)]
        libraries = ['openblas', 'openblas']
        library_dirs = ['/usr/local/lib']
    """
    ...
