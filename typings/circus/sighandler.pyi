"""
This type stub file was generated by pyright.
"""

class SysHandler:
    _SIGNALS_NAMES = ...
    SIGNALS = ...
    SIG_NAMES = ...
    def __init__(self, controller) -> None:
        ...
    
    def stop(self): # -> None:
        ...
    
    def signal(self, sig, frame=...):
        ...
    
    def quit(self): # -> None:
        ...
    
    def reload(self): # -> None:
        ...
    
    def handle_int(self): # -> None:
        ...
    
    def handle_term(self): # -> None:
        ...
    
    def handle_quit(self): # -> None:
        ...
    
    def handle_ill(self): # -> None:
        ...
    
    def handle_abrt(self): # -> None:
        ...
    
    def handle_break(self): # -> None:
        ...
    
    def handle_winch(self): # -> None:
        ...
    
    def handle_hup(self): # -> None:
        ...
    


