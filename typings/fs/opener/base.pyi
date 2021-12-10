import abc, typing
from typing import List, Text
import six
from ..base import FS
from .parse import ParseResult

if typing.TYPE_CHECKING: ...

@six.add_metaclass(abc.ABCMeta)
class Opener:
    protocols: List[Text] = ...
    def __repr__(self) -> Text: ...
    @abc.abstractmethod
    def open_fs(
        self,
        fs_url: Text,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: Text,
    ) -> FS: ...
