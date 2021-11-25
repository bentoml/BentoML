import sys
from distutils.command.sdist import sdist as old_sdist

from setuptools.command.sdist import sdist as old_sdist

if "setuptools" in sys.modules: ...
else: ...

class sdist(old_sdist):
    def add_defaults(self): ...
