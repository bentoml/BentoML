import sys

if sys.platform == "win32": ...
else:
    _Popen = object
if sys.version_info[:2] < (3, 3): ...
__all__ = ["Popen"]
TERMINATE = ...
WINEXE = ...
WINSERVICE = ...
WINENV = ...

class Popen(_Popen):
    method = ...
    def __init__(self, process_obj) -> None: ...
    def duplicate_for_child(self, handle): ...

def get_command_line(pipe_handle, **kwds): ...
def is_forking(argv): ...
def main(): ...
