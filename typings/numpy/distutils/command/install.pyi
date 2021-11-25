import distutils.command.install as old_install_mod
import sys

import setuptools.command.install as old_install_mod

if "setuptools" in sys.modules:
    have_setuptools = ...
else:
    have_setuptools = ...
old_install = old_install_mod.install

class install(old_install):
    sub_commands = old_install.sub_commands + [("install_clib", lambda x: True)]
    def finalize_options(self): ...
    def setuptools_run(self):
        """The setuptools version of the .run() method.

        We must pull in the entire code so we can override the level used in the
        _getframe() call since we wrap this call by one more level.
        """
        ...
    def run(self): ...
