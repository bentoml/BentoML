

from abc import ABC, abstractmethod
from argparse import ArgumentParser

class BaseHuggingfaceCLICommand(ABC):
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        ...
    
    @abstractmethod
    def run(self):
        ...
    


