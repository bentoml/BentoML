from contextlib import contextmanager
from typing import Iterator, Sequence, overload

from pandas._typing import DtypeArg
from pandas.core.api import DataFrame
from pandas.core.base import PandasObject

"""
Collection of query wrappers / abstractions to both facilitate data
retrieval and to reduce dependency on DB-specific API.
"""

class SQLAlchemyRequired(ImportError): ...
class DatabaseError(IOError): ...

_SQLALCHEMY_INSTALLED: bool | None = ...

def execute(sql, con, cur=..., params=...):
    """
    Execute the given SQL query using the provided connection object.

    Parameters
    ----------
    sql : string
        SQL query to be executed.
    con : SQLAlchemy connectable(engine/connection) or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by the
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    cur : deprecated, cursor is obtained from connection, default: None
    params : list or tuple, optional, default: None
        List of parameters to pass to execute method.

    Returns
    -------
    Results Iterable
    """
    ...

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
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a SQLAlchemy connectable, returns a DataFrame.
    This function does not support DBAPI connections.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : SQLAlchemy connectable or str
        A database URI could be provided as str.
        SQLite DBAPI connection mode not supported.
    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Can result in loss of Precision.
    parse_dates : list or dict, default None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default None
        List of column names to select from SQL table.
    chunksize : int, default None
        If specified, returns an iterator where `chunksize` is the number of
        rows to include in each chunk.

    Returns
    -------
    DataFrame or Iterator[DataFrame]
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information will be converted to UTC.

    Examples
    --------
    >>> pd.read_sql_table('table_name', 'postgres:///db_name')  # doctest:+SKIP
    """
    ...

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
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default integer index will be used.

    Parameters
    ----------
    sql : str SQL query or SQLAlchemy Selectable (select or text object)
        SQL query to be executed.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point. Useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the number of
        rows to include in each chunk.
    dtype : Type name or dict of columns
        Data type for data or columns. E.g. np.float64 or
        {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}

        .. versionadded:: 1.3.0

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Notes
    -----
    Any datetime values with time zone information parsed via the `parse_dates`
    parameter will be converted to UTC.
    """
    ...

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
) -> DataFrame | Iterator[DataFrame]:
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    Parameters
    ----------
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable, str, or sqlite3 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
        for engine disposal and connection closure for the SQLAlchemy connectable; str
        connections are closed automatically. See
        `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default True
        Attempts to convert values of non-string, non-numeric objects (like
        decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional, default: None
        List of parameters to pass to execute method.  The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
        Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}.
    parse_dates : list or dict, default: None
        - List of column names to parse as dates.
        - Dict of ``{column_name: format string}`` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
          to the keyword arguments of :func:`pandas.to_datetime`
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    chunksize : int, default None
        If specified, return an iterator where `chunksize` is the
        number of rows to include in each chunk.

    Returns
    -------
    DataFrame or Iterator[DataFrame]

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Examples
    --------
    Read data from SQL via either a SQL query or a SQL tablename.
    When using a SQLite database only SQL queries are accepted,
    providing only the SQL tablename will result in an error.

    >>> from sqlite3 import connect
    >>> conn = connect(':memory:')
    >>> df = pd.DataFrame(data=[[0, '10/11/12'], [1, '12/11/10']],
    ...                   columns=['int_column', 'date_column'])
    >>> df.to_sql('test_data', conn)

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data', conn)
       int_column date_column
    0           0    10/11/12
    1           1    12/11/10

    >>> pd.read_sql('test_data', 'postgres:///db_name')  # doctest:+SKIP

    Apply date parsing to columns through the ``parse_dates`` argument

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates=["date_column"])
       int_column date_column
    0           0  2012-10-11
    1           1  2010-12-11

    The ``parse_dates`` argument calls ``pd.to_datetime`` on the provided columns.
    Custom argument values for applying ``pd.to_datetime`` on a column are specified
    via a dictionary format:
    1. Ignore errors while parsing the values of "date_column"

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates={"date_column": {"errors": "ignore"}})
       int_column date_column
    0           0  2012-10-11
    1           1  2010-12-11

    2. Apply a dayfirst date parsing order on the values of "date_column"

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates={"date_column": {"dayfirst": True}})
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12

    3. Apply custom formatting when date parsing the values of "date_column"

    >>> pd.read_sql('SELECT int_column, date_column FROM test_data',
    ...             conn,
    ...             parse_dates={"date_column": {"format": "%d/%m/%y"}})
       int_column date_column
    0           0  2012-11-10
    1           1  2010-11-12
    """
    ...

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
) -> None:
    """
    Write records stored in a DataFrame to a SQL database.

    Parameters
    ----------
    frame : DataFrame, Series
    name : str
        Name of SQL table.
    con : SQLAlchemy connectable(engine/connection) or database string URI
        or sqlite3 DBAPI2 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : str, optional
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        - fail: If table exists, do nothing.
        - replace: If table exists, drop it, recreate it, and insert data.
        - append: If table exists, insert data. Create if does not exist.
    index : bool, default True
        Write DataFrame index as a column.
    index_label : str or sequence, optional
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 fallback mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        - None : Uses standard SQL ``INSERT`` clause (one per row).
        - 'multi': Pass multiple values in a single ``INSERT`` clause.
        - callable with signature ``(pd_table, conn, keys, data_iter)``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.
    engine : {'auto', 'sqlalchemy'}, default 'auto'
        SQL engine library to use. If 'auto', then the option
        ``io.sql.engine`` is used. The default ``io.sql.engine``
        behavior is 'sqlalchemy'

        .. versionadded:: 1.3.0

    **engine_kwargs
        Any additional kwargs are passed to the engine.
    """
    ...

def has_table(table_name: str, con, schema: str | None = ...):  # -> bool:
    """
    Check if DataBase has named table.

    Parameters
    ----------
    table_name: string
        Name of SQL table.
    con: SQLAlchemy connectable(engine/connection) or sqlite3 DBAPI2 connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library.
        If a DBAPI2 object, only sqlite3 is supported.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor supports
        this). If None, use default schema (default).

    Returns
    -------
    boolean
    """
    ...

table_exists = ...

def pandasSQL_builder(
    con, schema: str | None = ..., meta=..., is_cursor: bool = ...
):  # -> SQLDatabase | SQLiteDatabase:
    """
    Convenience function to return the correct PandasSQL subclass based on the
    provided parameters.
    """
    ...

class SQLTable(PandasObject):
    """
    For mapping Pandas tables to SQL tables.
    Uses fact that table is reflected by SQLAlchemy to
    do better type conversions.
    Also holds various flags needed to avoid having to
    pass them between functions all the time.
    """

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
    """
    Subclasses Should define read_sql and to_sql.
    """

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
    ):  # -> NoReturn:
        """
        Inserts data into already-prepared table
        """
        ...

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

def get_engine(engine: str) -> BaseEngine:
    """return our implementation"""
    ...

class SQLDatabase(PandasSQL):
    """
    This class enables conversion between DataFrame and SQL databases
    using SQLAlchemy to handle DataBase abstraction.

    Parameters
    ----------
    engine : SQLAlchemy connectable
        Connectable to connect with the database. Using SQLAlchemy makes it
        possible to use any DB supported by that library.
    schema : string, default None
        Name of SQL schema in database to write to (if database flavor
        supports this). If None, use default schema (default).
    meta : SQLAlchemy MetaData object, default None
        If provided, this MetaData object is used instead of a newly
        created. This allows to specify database flavor specific
        arguments in the MetaData object.

    """

    def __init__(self, engine, schema: str | None = ..., meta=...) -> None: ...
    @contextmanager
    def run_transaction(self): ...
    def execute(self, *args, **kwargs):
        """Simple passthrough to SQLAlchemy connectable"""
        ...
    def read_table(
        self,
        table_name: str,
        index_col: str | Sequence[str] | None = ...,
        coerce_float: bool = ...,
        parse_dates=...,
        columns=...,
        schema: str | None = ...,
        chunksize: int | None = ...,
    ):  # -> Generator[DataFrame, None, None] | DataFrame:
        """
        Read SQL database table into a DataFrame.

        Parameters
        ----------
        table_name : str
            Name of SQL table in database.
        index_col : string, optional, default: None
            Column to set as index.
        coerce_float : bool, default True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point. This can result in
            loss of precision.
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg}``, where the arg corresponds
              to the keyword arguments of :func:`pandas.to_datetime`.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, default: None
            List of column names to select from SQL table.
        schema : string, default None
            Name of SQL schema in database to query (if database flavor
            supports this).  If specified, this overwrites the default
            schema of the SQL database object.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.

        Returns
        -------
        DataFrame

        See Also
        --------
        pandas.read_sql_table
        SQLDatabase.read_query

        """
        ...
    def read_query(
        self,
        sql: str,
        index_col: str | None = ...,
        coerce_float: bool = ...,
        parse_dates=...,
        params=...,
        chunksize: int | None = ...,
        dtype: DtypeArg | None = ...,
    ):  # -> Generator[DataFrame, None, None] | DataFrame:
        """
        Read SQL query into a DataFrame.

        Parameters
        ----------
        sql : str
            SQL query to be executed.
        index_col : string, optional, default: None
            Column name to use as index for the returned DataFrame object.
        coerce_float : bool, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional, default: None
            List of parameters to pass to execute method.  The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
            Eg. for psycopg2, uses %(name)s so use params={'name' : 'value'}
        parse_dates : list or dict, default: None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime` Especially useful with databases
              without native Datetime support, such as SQLite.
        chunksize : int, default None
            If specified, return an iterator where `chunksize` is the number
            of rows to include in each chunk.
        dtype : Type name or dict of columns
            Data type for data or columns. E.g. np.float64 or
            {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}

            .. versionadded:: 1.3.0

        Returns
        -------
        DataFrame

        See Also
        --------
        read_sql_table : Read SQL database table into a DataFrame.
        read_sql

        """
        ...
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
    ) -> SQLTable:
        """
        Prepares table in the database for data insertion. Creates it if needed, etc.
        """
        ...
    def check_case_sensitive(self, name, schema):  # -> None:
        """
        Checks table name for issues with case-sensitivity.
        Method is called after data is inserted.
        """
        ...
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
    ):  # -> None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame : DataFrame
        name : string
            Name of SQL table.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            - fail: If table exists, do nothing.
            - replace: If table exists, drop it, recreate it, and insert data.
            - append: If table exists, insert data. Create if does not exist.
        index : boolean, default True
            Write DataFrame index as a column.
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Name of SQL schema in database to write to (if database flavor
            supports this). If specified, this overwrites the default
            schema of the SQLDatabase object.
        chunksize : int, default None
            If not None, then rows will be written in batches of this size at a
            time.  If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a SQLAlchemy type. If all columns are of the same type, one
            single value can be used.
        method : {None', 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        engine : {'auto', 'sqlalchemy'}, default 'auto'
            SQL engine library to use. If 'auto', then the option
            ``io.sql.engine`` is used. The default ``io.sql.engine``
            behavior is 'sqlalchemy'

            .. versionadded:: 1.3.0

        **engine_kwargs
            Any additional kwargs are passed to the engine.
        """
        ...
    @property
    def tables(self): ...
    def has_table(self, name: str, schema: str | None = ...): ...
    def get_table(self, table_name: str, schema: str | None = ...): ...
    def drop_table(self, table_name: str, schema: str | None = ...): ...

_SQL_TYPES = ...
_SAFE_NAMES_WARNING = ...

class SQLiteTable(SQLTable):
    """
    Patch the SQLTable for fallback support.
    Instead of a table variable just use the Create Table statement.
    """

    def __init__(self, *args, **kwargs) -> None: ...
    def sql_schema(self): ...
    def insert_statement(self, *, num_rows: int): ...

class SQLiteDatabase(PandasSQL):
    """
    Version of SQLDatabase to support SQLite connections (fallback without
    SQLAlchemy). This should only be used internally.

    Parameters
    ----------
    con : sqlite connection object

    """

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
    ):  # -> None:
        """
        Write records stored in a DataFrame to a SQL database.

        Parameters
        ----------
        frame: DataFrame
        name: string
            Name of SQL table.
        if_exists: {'fail', 'replace', 'append'}, default 'fail'
            fail: If table exists, do nothing.
            replace: If table exists, drop it, recreate it, and insert data.
            append: If table exists, insert data. Create if it does not exist.
        index : bool, default True
            Write DataFrame index as a column
        index_label : string or sequence, default None
            Column label for index column(s). If None is given (default) and
            `index` is True, then the index names are used.
            A sequence should be given if the DataFrame uses MultiIndex.
        schema : string, default None
            Ignored parameter included for compatibility with SQLAlchemy
            version of ``to_sql``.
        chunksize : int, default None
            If not None, then rows will be written in batches of this
            size at a time. If None, all rows will be written at once.
        dtype : single type or dict of column name to SQL type, default None
            Optional specifying the datatype for columns. The SQL type should
            be a string. If all columns are of the same type, one single value
            can be used.
        method : {None, 'multi', callable}, default None
            Controls the SQL insertion clause used:

            * None : Uses standard SQL ``INSERT`` clause (one per row).
            * 'multi': Pass multiple values in a single ``INSERT`` clause.
            * callable with signature ``(pd_table, conn, keys, data_iter)``.

            Details and a sample callable implementation can be found in the
            section :ref:`insert method <io.sql.method>`.
        """
        ...
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
):  # -> str:
    """
    Get the SQL db table schema for the given frame.

    Parameters
    ----------
    frame : DataFrame
    name : str
        name of SQL table
    keys : string or sequence, default: None
        columns to use a primary key
    con: an open SQL database connection object or a SQLAlchemy connectable
        Using SQLAlchemy makes it possible to use any DB supported by that
        library, default: None
        If a DBAPI2 object, only sqlite3 is supported.
    dtype : dict of column name to SQL type, default None
        Optional specifying the datatype for columns. The SQL type should
        be a SQLAlchemy type, or a string for sqlite3 fallback connection.
    schema: str, default: None
        Optional specifying the schema to be used in creating the table.

        .. versionadded:: 1.2.0
    """
    ...
