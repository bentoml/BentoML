from distutils.command.build_scripts import build_scripts as old_build_scripts

""" Modified version of build_scripts that handles building scripts from functions.

"""

class build_scripts(old_build_scripts):
    def generate_scripts(self, scripts): ...
    def run(self): ...
    def get_source_files(self): ...
