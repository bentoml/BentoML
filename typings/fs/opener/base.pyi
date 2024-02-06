import abc
from typing import List
from typing import Text

import six

from ..base import FS
from .parse import ParseResult

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
