from setuptools.command.develop import develop as old_develop

""" Override the develop command from setuptools so we can ensure that our
generated files (from build_src or build_scripts) are properly converted to real
files with filenames.

"""

class develop(old_develop):
    __doc__ = ...
    def install_for_development(self): ...
