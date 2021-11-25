from distutils.command.config import config as old_config

class config(old_config):
    def initialize_options(self): ...
    def check_header(self, header, include_dirs=..., library_dirs=..., lang=...): ...
    def check_decl(self, symbol, headers=..., include_dirs=...): ...
    def check_macro_true(self, symbol, headers=..., include_dirs=...): ...
    def check_type(self, type_name, headers=..., include_dirs=..., library_dirs=...):
        """Check type availability. Return True if the type can be compiled,
        False otherwise"""
        ...
    def check_type_size(
        self, type_name, headers=..., include_dirs=..., library_dirs=..., expected=...
    ):
        """Check size of a given type."""
        ...
    def check_func(
        self,
        func,
        headers=...,
        include_dirs=...,
        libraries=...,
        library_dirs=...,
        decl=...,
        call=...,
        call_args=...,
    ): ...
    def check_funcs_once(
        self,
        funcs,
        headers=...,
        include_dirs=...,
        libraries=...,
        library_dirs=...,
        decl=...,
        call=...,
        call_args=...,
    ):
        """Check a list of functions at once.

        This is useful to speed up things, since all the functions in the funcs
        list will be put in one compilation unit.

        Arguments
        ---------
        funcs : seq
            list of functions to test
        include_dirs : seq
            list of header paths
        libraries : seq
            list of libraries to link the code snippet to
        library_dirs : seq
            list of library paths
        decl : dict
            for every (key, value), the declaration in the value will be
            used for function in key. If a function is not in the
            dictionary, no declaration will be used.
        call : dict
            for every item (f, value), if the value is True, a call will be
            done to the function f.
        """
        ...
    def check_inline(self):
        """Return the inline keyword recognized by the compiler, empty string
        otherwise."""
        ...
    def check_restrict(self):
        """Return the restrict keyword recognized by the compiler, empty string
        otherwise."""
        ...
    def check_compiler_gcc(self):
        """Return True if the C compiler is gcc"""
        ...
    def check_gcc_function_attribute(self, attribute, name): ...
    def check_gcc_function_attribute_with_intrinsics(
        self, attribute, name, code, include
    ): ...
    def check_gcc_variable_attribute(self, attribute): ...
    def check_gcc_version_at_least(self, major, minor=..., patchlevel=...):
        """Return True if the GCC version is greater than or equal to the
        specified version."""
        ...
    def get_output(
        self,
        body,
        headers=...,
        include_dirs=...,
        libraries=...,
        library_dirs=...,
        lang=...,
        use_tee=...,
    ):
        """Try to compile, link to an executable, and run a program
        built from 'body' and 'headers'. Returns the exit status code
        of the program and its output.
        """
        ...

class GrabStdout:
    def __init__(self) -> None: ...
    def write(self, data): ...
    def flush(self): ...
    def restore(self): ...
