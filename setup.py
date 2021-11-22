import versioneer
from setuptools import setup

if __name__ == "__main__":
    setup(version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass())
