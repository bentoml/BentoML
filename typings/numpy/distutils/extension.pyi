from distutils.extension import Extension as old_Extension

"""distutils.extension

Provides the Extension class, used to describe C/C++ extension
modules in setup scripts.

Overridden to support f2py.

"""
cxx_ext_re = ...
fortran_pyf_ext_re = ...

class Extension(old_Extension):
    """
    Parameters
    ----------
    name : str
        Extension name.
    sources : list of str
        List of source file locations relative to the top directory of
        the package.
    extra_compile_args : list of str
        Extra command line arguments to pass to the compiler.
    extra_f77_compile_args : list of str
        Extra command line arguments to pass to the fortran77 compiler.
    extra_f90_compile_args : list of str
        Extra command line arguments to pass to the fortran90 compiler.
    """

    def __init__(
        self,
        name,
        sources,
        include_dirs=...,
        define_macros=...,
        undef_macros=...,
        library_dirs=...,
        libraries=...,
        runtime_library_dirs=...,
        extra_objects=...,
        extra_compile_args=...,
        extra_link_args=...,
        export_symbols=...,
        swig_opts=...,
        depends=...,
        language=...,
        f2py_options=...,
        module_dirs=...,
        extra_f77_compile_args=...,
        extra_f90_compile_args=...,
    ) -> None: ...
    def has_cxx_sources(self): ...
    def has_f2py_sources(self): ...
