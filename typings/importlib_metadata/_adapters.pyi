

class Message(email.message.Message):
    multiple_use_keys = ...
    def __new__(cls, orig: email.message.Message): # -> Self@Message:
        ...
    
    def __init__(self, *args, **kwargs) -> None:
        ...
    
    def __iter__(self): # -> Iterator[str]:
        ...
    
    @property
    def json(self): # -> dict[Unknown, Unknown | list[_HeaderType] | _T@get_all | _HeaderType]:
        """
        Convert PackageMetadata to a JSON-compatible format
        per PEP 0566.
        """
        ...
    


