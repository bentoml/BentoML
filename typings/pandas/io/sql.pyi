from contextlib import contextmanager
from typing import Iterator, Sequence, overload
from pandas._typing import DtypeArg
from pandas.core.api import DataFrame
from pandas.core.base import PandasObject

class SQLAlchemyRequired(ImportError): ...
class DatabaseError(IOError): ...

_SQLALCHEMY_INSTALLED: bool | None = ...

def execute(sql, con, cur=..., params=...): ...
@overload
def read_sql_table(
    table_name,
    con,
    schema=...,
    index_col=...,
    coerce_float=...,
    parse_dates=...,
    columns=...,
    chunksize: None = ...,
) -> DataFrame: ...
@overload
def read_sql_table(
    table_name,
    con,
    schema=...,
    index_col=...,
    coerce_float=...,
    parse_dates=...,
    columns=...,
    chunksize: int = ...,
) -> Iterator[DataFrame]: ...
def read_sql_table(
    table_name: str,
    con,
    schema: str | None = ...,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    parse_dates=...,
    columns=...,
    chunksize: int | None = ...,
) -> DataFrame | Iterator[DataFrame]: ...
@overload
def read_sql_query(
    sql,
    con,
    index_col=...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    chunksize: None = ...,
    dtype: DtypeArg | None = ...,
) -> DataFrame: ...
@overload
def read_sql_query(
    sql,
    con,
    index_col=...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    chunksize: int = ...,
    dtype: DtypeArg | None = ...,
) -> Iterator[DataFrame]: ...
def read_sql_query(
    sql,
    con,
    index_col=...,
    coerce_float: bool = ...,
    params=...,
    parse_dates=...,
    chunksize: int | None = ...,
    dtype: DtypeArg | None = ...,
) -> DataFrame | Iterator[DataFrame]: ...
@overload
def read_sql(
    sql,
    con,
    index_col=...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns=...,
    chunksize: None = ...,
) -> DataFrame: ...
@overload
def read_sql(
    sql,
    con,
    index_col=...,
    coerce_float=...,
    params=...,
    parse_dates=...,
    columns=...,
    chunksize: int = ...,
) -> Iterator[DataFrame]: ...
def read_sql(
    sql,
    con,
    index_col: str | Sequence[str] | None = ...,
    coerce_float: bool = ...,
    params=...,
    parse_dates=...,
    columns=...,
    chunksize: int | None = ...,
) -> DataFrame | Iterator[DataFrame]: ...
def to_sql(
    frame,
    name: str,
    con,
    schema: str | None = ...,
    if_exists: str = ...,
    index: bool = ...,
    index_label=...,
    chunksize: int | None = ...,
    dtype: DtypeArg | None = ...,
    method: str | None = ...,
    engine: str = ...,
    **engine_kwargs
) -> None: ...
def has_table(table_name: str, con, schema: str | None = ...): ...

table_exists = ...

def pandasSQL_builder(
    con, schema: str | None = ..., meta=..., is_cursor: bool = ...
): ...

class SQLTable(PandasObject):
    def __init__(
        self,
        name: str,
        pandas_sql_engine,
        frame=...,
        index=...,
        if_exists=...,
        prefix=...,
        index_label=...,
        schema=...,
        keys=...,
        dtype: DtypeArg | None = ...,
    ) -> None: ...
    def exists(self): ...
    def sql_schema(self): ...
    def create(self): ...
    def insert_data(self): ...
    def insert(self, chunksize: int | None = ..., method: str | None = ...): ...
    def read(self, coerce_float=..., parse_dates=..., columns=..., chunksize=...): ...

class PandasSQL(PandasObject):
    def read_sql(self, *args, **kwargs): ...
    def to_sql(
        self,
        frame,
        name,
        if_exists=...,
        index=...,
        index_label=...,
        schema=...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method=...,
    ): ...

class BaseEngine:
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name,
        index=...,
        schema=...,
        chunksize=...,
        method=...,
        **engine_kwargs
    ): ...

class SQLAlchemyEngine(BaseEngine):
    def __init__(self) -> None: ...
    def insert_records(
        self,
        table: SQLTable,
        con,
        frame,
        name,
        index=...,
        schema=...,
        chunksize=...,
        method=...,
        **engine_kwargs
    ): ...

def get_engine(engine: str) -> BaseEngine: ...

class SQLDatabase(PandasSQL):
    def __init__(self, engine, schema: str | None = ..., meta=...) -> None: ...
    @contextmanager
    def run_transaction(self): ...
    def execute(self, *args, **kwargs): ...
    def read_table(
        self,
        table_name: str,
        index_col: str | Sequence[str] | None = ...,
        coerce_float: bool = ...,
        parse_dates=...,
        columns=...,
        schema: str | None = ...,
        chunksize: int | None = ...,
    ): ...
    def read_query(
        self,
        sql: str,
        index_col: str | None = ...,
        coerce_float: bool = ...,
        parse_dates=...,
        params=...,
        chunksize: int | None = ...,
        dtype: DtypeArg | None = ...,
    ): ...
    read_sql = ...
    def prep_table(
        self,
        frame,
        name,
        if_exists=...,
        index=...,
        index_label=...,
        schema=...,
        dtype: DtypeArg | None = ...,
    ) -> SQLTable: ...
    def check_case_sensitive(self, name, schema): ...
    def to_sql(
        self,
        frame,
        name,
        if_exists=...,
        index=...,
        index_label=...,
        schema=...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method=...,
        engine=...,
        **engine_kwargs
    ): ...
    @property
    def tables(self): ...
    def has_table(self, name: str, schema: str | None = ...): ...
    def get_table(self, table_name: str, schema: str | None = ...): ...
    def drop_table(self, table_name: str, schema: str | None = ...): ...

_SQL_TYPES = ...
_SAFE_NAMES_WARNING = ...

class SQLiteTable(SQLTable):
    def __init__(self, *args, **kwargs) -> None: ...
    def sql_schema(self): ...
    def insert_statement(self, *, num_rows: int): ...

class SQLiteDatabase(PandasSQL):
    def __init__(self, con, is_cursor: bool = ...) -> None: ...
    @contextmanager
    def run_transaction(self): ...
    def execute(self, *args, **kwargs): ...
    def read_query(
        self,
        sql,
        index_col=...,
        coerce_float: bool = ...,
        params=...,
        parse_dates=...,
        chunksize: int | None = ...,
        dtype: DtypeArg | None = ...,
    ): ...
    def to_sql(
        self,
        frame,
        name,
        if_exists=...,
        index=...,
        index_label=...,
        schema=...,
        chunksize=...,
        dtype: DtypeArg | None = ...,
        method=...,
        **kwargs
    ): ...
    def has_table(self, name: str, schema: str | None = ...): ...
    def get_table(self, table_name: str, schema: str | None = ...): ...
    def drop_table(self, name: str, schema: str | None = ...): ...

def get_schema(
    frame,
    name: str,
    keys=...,
    con=...,
    dtype: DtypeArg | None = ...,
    schema: str | None = ...,
): ...
