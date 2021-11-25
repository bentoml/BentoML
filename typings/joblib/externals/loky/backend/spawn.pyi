import sys

if sys.platform != "win32":
    WINEXE = ...
    WINSERVICE = ...
else: ...
if WINSERVICE:
    _python_exe = ...
else:
    _python_exe = ...

def get_executable(): ...
def get_preparation_data(name, init_main_module=...):
    """
    Return info about parent needed by child to unpickle process object
    """
    ...

old_main_modules = ...

def prepare(data):  # -> None:
    """
    Try to get current process ready to unpickle process object
    """
    ...

def import_main_path(main_path):  # -> None:
    """
    Set sys.modules['__main__'] to module at main_path
    """
    ...
