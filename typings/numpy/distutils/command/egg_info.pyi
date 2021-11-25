from setuptools.command.egg_info import egg_info as _egg_info

class egg_info(_egg_info):
    def run(self): ...
