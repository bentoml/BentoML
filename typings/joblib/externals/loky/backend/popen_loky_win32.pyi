"""
This type stub file was generated by pyright.
"""

import sys

if sys.platform == "win32":
    ...
else:
    _Popen = object
if sys.version_info[: 2] < (3, 3):
    ...
__all__ = ['Popen']
TERMINATE = ...
WINEXE = ...
WINSERVICE = ...
WINENV = ...
class Popen(_Popen):
    '''
    Start a subprocess to run the code of a process object
    '''
    method = ...
    def __init__(self, process_obj) -> None:
        ...
    
    def duplicate_for_child(self, handle):
        ...
    


def get_command_line(pipe_handle, **kwds): # -> list[str | Unknown]:
    '''
    Returns prefix of command line used for spawning a child process
    '''
    ...

def is_forking(argv): # -> bool:
    '''
    Return whether commandline indicates we are forking
    '''
    ...

def main(): # -> NoReturn:
    '''
    Run code specified by data received over pipe
    '''
    ...

