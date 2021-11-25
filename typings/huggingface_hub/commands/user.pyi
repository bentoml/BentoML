

from argparse import ArgumentParser
from typing import List, Union

from huggingface_hub.commands import BaseHuggingfaceCLICommand

class UserCommands(BaseHuggingfaceCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser): # -> None:
        ...
    


class ANSI:
    """
    Helper for en.wikipedia.org/wiki/ANSI_escape_code
    """
    _bold = ...
    _red = ...
    _gray = ...
    _reset = ...
    @classmethod
    def bold(cls, s): # -> str:
        ...
    
    @classmethod
    def red(cls, s): # -> str:
        ...
    
    @classmethod
    def gray(cls, s): # -> str:
        ...
    


def tabulate(rows: List[List[Union[str, int]]], headers: List[str]) -> str:
    """
    Inspired by:

    - stackoverflow.com/a/8356620/593036
    - stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
    """
    ...

def currently_setup_credential_helpers(directory=...) -> List[str]:
    ...

class BaseUserCommand:
    def __init__(self, args) -> None:
        ...
    


class LoginCommand(BaseUserCommand):
    def run(self): # -> None:
        ...
    


class WhoamiCommand(BaseUserCommand):
    def run(self): # -> None:
        ...
    


class LogoutCommand(BaseUserCommand):
    def run(self): # -> None:
        ...
    


class ListReposObjsCommand(BaseUserCommand):
    def run(self): # -> None:
        ...
    


class RepoCreateCommand(BaseUserCommand):
    def run(self): # -> None:
        ...
    


LOGIN_NOTEBOOK_HTML = ...
def notebook_login(): # -> None:
    """
    Displays a widget to login to the HF website and store the token.
    """
    ...

