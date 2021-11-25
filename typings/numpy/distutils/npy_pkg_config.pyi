__all__ = [
    "FormatError",
    "PkgNotFound",
    "LibraryInfo",
    "VariableSet",
    "read_config",
    "parse_flags",
]
_VAR = ...

class FormatError(IOError):
    """
    Exception thrown when there is a problem parsing a configuration file.

    """

    def __init__(self, msg) -> None: ...
    def __str__(self) -> str: ...

class PkgNotFound(IOError):
    """Exception raised when a package can not be located."""

    def __init__(self, msg) -> None: ...
    def __str__(self) -> str: ...

def parse_flags(line):
    """
    Parse a line from a config file containing compile flags.

    Parameters
    ----------
    line : str
        A single line containing one or more compile flags.

    Returns
    -------
    d : dict
        Dictionary of parsed flags, split into relevant categories.
        These categories are the keys of `d`:

        * 'include_dirs'
        * 'library_dirs'
        * 'libraries'
        * 'macros'
        * 'ignored'

    """
    ...

class LibraryInfo:
    """
    Object containing build information about a library.

    Parameters
    ----------
    name : str
        The library name.
    description : str
        Description of the library.
    version : str
        Version string.
    sections : dict
        The sections of the configuration file for the library. The keys are
        the section headers, the values the text under each header.
    vars : class instance
        A `VariableSet` instance, which contains ``(name, value)`` pairs for
        variables defined in the configuration file for the library.
    requires : sequence, optional
        The required libraries for the library to be installed.

    Notes
    -----
    All input parameters (except "sections" which is a method) are available as
    attributes of the same name.

    """

    def __init__(
        self, name, description, version, sections, vars, requires=...
    ) -> None: ...
    def sections(self):
        """
        Return the section headers of the config file.

        Parameters
        ----------
        None

        Returns
        -------
        keys : list of str
            The list of section headers.

        """
        ...
    def cflags(self, section=...): ...
    def libs(self, section=...): ...
    def __str__(self) -> str: ...

class VariableSet:
    """
    Container object for the variables defined in a config file.

    `VariableSet` can be used as a plain dictionary, with the variable names
    as keys.

    Parameters
    ----------
    d : dict
        Dict of items in the "variables" section of the configuration file.

    """

    def __init__(self, d) -> None: ...
    def interpolate(self, value): ...
    def variables(self):
        """
        Return the list of variable names.

        Parameters
        ----------
        None

        Returns
        -------
        names : list of str
            The names of all variables in the `VariableSet` instance.

        """
        ...
    def __getitem__(self, name): ...
    def __setitem__(self, name, value): ...

def parse_meta(config): ...
def parse_variables(config): ...
def parse_sections(config): ...
def pkg_to_filename(pkg_name): ...
def parse_config(filename, dirs=...): ...

_CACHE = ...

def read_config(pkgname, dirs=...):
    """
    Return library info for a package from its configuration file.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of directories - usually including
        the NumPy base directory - where to look for npy-pkg-config files.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    misc_util.get_info, misc_util.get_pkg_info

    Examples
    --------
    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
    >>> type(npymath_info)
    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
    >>> print(npymath_info)
    Name: npymath
    Description: Portable, core math library implementing C99 standard
    Requires:
    Version: 0.1  #random

    """
    ...

if __name__ == "__main__":
    parser = ...
    pkg_name = ...
    d = ...
