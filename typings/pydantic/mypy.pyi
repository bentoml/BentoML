"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from typing import Type as TypingType
from typing import Union

from mypy.nodes import (Argument, AssignmentStmt, ClassDef, Context, FuncBase,
                        JsonDict, NameExpr, SymbolNode, TypeInfo, Var)
from mypy.options import Options
from mypy.plugin import (CheckerPluginInterface, ClassDefContext,
                         MethodContext, Plugin,
                         SemanticAnalyzerPluginInterface)
from mypy.types import Type, TypeVarDef

CONFIGFILE_KEY = ...
METADATA_KEY = ...
BASEMODEL_FULLNAME = ...
BASESETTINGS_FULLNAME = ...
FIELD_FULLNAME = ...
DATACLASS_FULLNAME = ...

def plugin(version: str) -> TypingType[Plugin]:
    """
    `version` is the mypy version string

    We might want to use this to print a warning if the mypy version being used is
    newer, or especially older, than we expect (or need).
    """
    ...

class PydanticPlugin(Plugin):
    def __init__(self, options: Options) -> None: ...
    def get_base_class_hook(
        self, fullname: str
    ) -> Optional[Callable[[ClassDefContext], None]]: ...
    def get_method_hook(
        self, fullname: str
    ) -> Optional[Callable[[MethodContext], Type]]: ...
    def get_class_decorator_hook(
        self, fullname: str
    ) -> Optional[Callable[[ClassDefContext], None]]: ...

class PydanticPluginConfig:
    __slots__ = ...
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    warn_untyped_fields: bool
    def __init__(self, options: Options) -> None: ...

def from_orm_callback(ctx: MethodContext) -> Type:
    """
    Raise an error if orm_mode is not enabled
    """
    ...

class PydanticModelTransformer:
    tracked_config_fields: Set[str] = ...
    def __init__(
        self, ctx: ClassDefContext, plugin_config: PydanticPluginConfig
    ) -> None: ...
    def transform(self) -> None:
        """
        Configures the BaseModel subclass according to the plugin settings.

        In particular:
        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if allow_mutation = False or frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        ...
    def collect_config(self) -> ModelConfigData:
        """
        Collects the values of the config attributes that are used by the plugin, accounting for parent classes.
        """
        ...
    def collect_fields(self, model_config: ModelConfigData) -> List[PydanticModelField]:
        """
        Collects the fields for the model, accounting for parent classes
        """
        ...
    def add_initializer(
        self,
        fields: List[PydanticModelField],
        config: ModelConfigData,
        is_settings: bool,
    ) -> None:
        """
        Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
        ...
    def add_construct_method(self, fields: List[PydanticModelField]) -> None:
        """
        Adds a fully typed `construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
        ...
    def set_frozen(self, fields: List[PydanticModelField], frozen: bool) -> None:
        """
        Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
        ...
    def get_config_update(self, substmt: AssignmentStmt) -> Optional[ModelConfigData]:
        """
        Determines the config update due to a single statement in the Config class definition.

        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)
        """
        ...
    @staticmethod
    def get_is_required(cls: ClassDef, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
        """
        Returns a boolean indicating whether the field defined in `stmt` is a required field.
        """
        ...
    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
        """
        Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
        ...
    def get_field_arguments(
        self,
        fields: List[PydanticModelField],
        typed: bool,
        force_all_optional: bool,
        use_alias: bool,
    ) -> List[Argument]:
        """
        Helper function used during the construction of the `__init__` and `construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
        ...
    def should_init_forbid_extra(
        self, fields: List[PydanticModelField], config: ModelConfigData
    ) -> bool:
        """
        Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature

        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,
        *unless* a required dynamic alias is present (since then we can't determine a valid signature).
        """
        ...
    @staticmethod
    def is_dynamic_alias_present(
        fields: List[PydanticModelField], has_alias_generator: bool
    ) -> bool:
        """
        Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be
        determined during static analysis.
        """
        ...

class PydanticModelField:
    def __init__(
        self,
        name: str,
        is_required: bool,
        alias: Optional[str],
        has_dynamic_alias: bool,
        line: int,
        column: int,
    ) -> None: ...
    def to_var(self, info: TypeInfo, use_alias: bool) -> Var: ...
    def to_argument(
        self, info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool
    ) -> Argument: ...
    def serialize(self) -> JsonDict: ...
    @classmethod
    def deserialize(cls, info: TypeInfo, data: JsonDict) -> PydanticModelField: ...

class ModelConfigData:
    def __init__(
        self,
        forbid_extra: Optional[bool] = ...,
        allow_mutation: Optional[bool] = ...,
        frozen: Optional[bool] = ...,
        orm_mode: Optional[bool] = ...,
        allow_population_by_field_name: Optional[bool] = ...,
        has_alias_generator: Optional[bool] = ...,
    ) -> None: ...
    def set_values_dict(self) -> Dict[str, Any]: ...
    def update(self, config: Optional[ModelConfigData]) -> None: ...
    def setdefault(self, key: str, value: Any) -> None: ...

ERROR_ORM = ...
ERROR_CONFIG = ...
ERROR_ALIAS = ...
ERROR_UNEXPECTED = ...
ERROR_UNTYPED = ...

def error_from_orm(
    model_name: str, api: CheckerPluginInterface, context: Context
) -> None: ...
def error_invalid_config_value(
    name: str, api: SemanticAnalyzerPluginInterface, context: Context
) -> None: ...
def error_required_dynamic_aliases(
    api: SemanticAnalyzerPluginInterface, context: Context
) -> None: ...
def error_unexpected_behavior(
    detail: str, api: CheckerPluginInterface, context: Context
) -> None: ...
def error_untyped_fields(
    api: SemanticAnalyzerPluginInterface, context: Context
) -> None: ...
def add_method(
    ctx: ClassDefContext,
    name: str,
    args: List[Argument],
    return_type: Type,
    self_type: Optional[Type] = ...,
    tvar_def: Optional[TypeVarDef] = ...,
    is_classmethod: bool = ...,
    is_new: bool = ...,
) -> None:
    """
    Adds a new method to a class.

    This can be dropped if/when https://github.com/python/mypy/issues/7301 is merged
    """
    ...

def get_fullname(x: Union[FuncBase, SymbolNode]) -> str:
    """
    Used for compatibility with mypy 0.740; can be dropped once support for 0.740 is dropped.
    """
    ...

def get_name(x: Union[FuncBase, SymbolNode]) -> str:
    """
    Used for compatibility with mypy 0.740; can be dropped once support for 0.740 is dropped.
    """
    ...
