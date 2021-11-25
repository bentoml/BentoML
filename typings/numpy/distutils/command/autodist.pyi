"""This module implements additional tests ala autoconf which can be useful.

"""

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    ...

def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    ...

def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""
    ...

def check_gcc_version_at_least(cmd, major, minor=..., patchlevel=...):
    """
    Check that the gcc version is at least the specified version."""
    ...

def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    ...

def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code, include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    ...

def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    ...
