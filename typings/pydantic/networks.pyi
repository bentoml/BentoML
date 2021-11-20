"""
This type stub file was generated by pyright.
"""

from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
    _BaseAddress,
    _BaseNetwork,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Generator,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    Union,
    no_type_check,
)

from .fields import ModelField
from .main import BaseConfig
from .typing import AnyCallable
from .utils import Representation

if TYPE_CHECKING:
    CallableGenerator = Generator[AnyCallable, None, None]
else: ...
NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]]
__all__ = [
    "AnyUrl",
    "AnyHttpUrl",
    "HttpUrl",
    "stricturl",
    "EmailStr",
    "NameEmail",
    "IPvAnyAddress",
    "IPvAnyInterface",
    "IPvAnyNetwork",
    "PostgresDsn",
    "RedisDsn",
    "validate_email",
]
_url_regex_cache = ...
_ascii_domain_regex_cache = ...
_int_domain_regex_cache = ...

def url_regex() -> Pattern[str]: ...
def ascii_domain_regex() -> Pattern[str]: ...
def int_domain_regex() -> Pattern[str]: ...

class AnyUrl(str):
    strip_whitespace = ...
    min_length = ...
    max_length = 2 ** 16
    allowed_schemes: Optional[Set[str]] = ...
    tld_required: bool = ...
    user_required: bool = ...
    __slots__ = ...
    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs) -> object: ...
    def __init__(
        self,
        url: str,
        *,
        scheme: str,
        user: Optional[str] = ...,
        password: Optional[str] = ...,
        host: str,
        tld: Optional[str] = ...,
        host_type: str = ...,
        port: Optional[str] = ...,
        path: Optional[str] = ...,
        query: Optional[str] = ...,
        fragment: Optional[str] = ...
    ) -> None: ...
    @classmethod
    def build(
        cls,
        *,
        scheme: str,
        user: Optional[str] = ...,
        password: Optional[str] = ...,
        host: str,
        port: Optional[str] = ...,
        path: Optional[str] = ...,
        query: Optional[str] = ...,
        fragment: Optional[str] = ...,
        **kwargs: str
    ) -> str: ...
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls, value: Any, field: ModelField, config: BaseConfig) -> AnyUrl: ...
    @classmethod
    def validate_parts(cls, parts: Dict[str, str]) -> Dict[str, str]:
        """
        A method used to validate parts of an URL.
        Could be overridden to set default values for parts if missing
        """
        ...
    @classmethod
    def validate_host(
        cls, parts: Dict[str, str]
    ) -> Tuple[str, Optional[str], str, bool]: ...
    def __repr__(self) -> str: ...

class AnyHttpUrl(AnyUrl):
    allowed_schemes = ...

class HttpUrl(AnyUrl):
    allowed_schemes = ...
    tld_required = ...
    max_length = ...

class PostgresDsn(AnyUrl):
    allowed_schemes = ...
    user_required = ...

class RedisDsn(AnyUrl):
    allowed_schemes = ...
    @classmethod
    def validate_parts(cls, parts: Dict[str, str]) -> Dict[str, str]: ...

def stricturl(
    *,
    strip_whitespace: bool = ...,
    min_length: int = ...,
    max_length: int = ...,
    tld_required: bool = ...,
    allowed_schemes: Optional[Union[FrozenSet[str], Set[str]]] = ...
) -> Type[AnyUrl]: ...
def import_email_validator() -> None: ...

class EmailStr(str):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls, value: Union[str]) -> str: ...

class NameEmail(Representation):
    __slots__ = ...
    def __init__(self, name: str, email: str) -> None: ...
    def __eq__(self, other: Any) -> bool: ...
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls, value: Any) -> NameEmail: ...
    def __str__(self) -> str: ...

class IPvAnyAddress(_BaseAddress):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(
        cls, value: Union[str, bytes, int]
    ) -> Union[IPv4Address, IPv6Address]: ...

class IPvAnyInterface(_BaseAddress):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls, value: NetworkType) -> Union[IPv4Interface, IPv6Interface]: ...

class IPvAnyNetwork(_BaseNetwork):
    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None: ...
    @classmethod
    def __get_validators__(cls) -> CallableGenerator: ...
    @classmethod
    def validate(cls, value: NetworkType) -> Union[IPv4Network, IPv6Network]: ...

pretty_email_regex = ...

def validate_email(value: Union[str]) -> Tuple[str, str]:
    """
    Brutally simple email address validation. Note unlike most email address validation
    * raw ip address (literal) domain parts are not allowed.
    * "John Doe <local_part@domain.com>" style "pretty" email addresses are processed
    * the local part check is extremely basic. This raises the possibility of unicode spoofing, but no better
        solution is really possible.
    * spaces are striped from the beginning and end of addresses but no error is raised

    See RFC 5322 but treat it with suspicion, there seems to exist no universally acknowledged test for a valid email!
    """
    ...
