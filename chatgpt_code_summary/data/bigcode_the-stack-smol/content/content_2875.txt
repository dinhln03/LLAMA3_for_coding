import logging
import time
from abc import abstractmethod
from enum import Enum
from typing import Dict, Callable, Any, List

from schema import Schema

import sqlalchemy
from sqlalchemy.engine import ResultProxy
from sqlalchemy.orm import Query
from sqlalchemy.schema import Table
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.base import Connection
from contextlib import contextmanager

from flask_app.utilities.DataInterfaces import ConnectionOptions

logger = logging.getLogger(__name__)


class SqlDialect(Enum):
    postgres = "postgres"
    sqlite = "sqlite"

    @classmethod
    def has_value(cls, value) -> bool:
        return any(value == item.value for item in cls)


# TODO: Connection Factory
class SqlConnectionOptions(ConnectionOptions):
    @staticmethod
    def factory(sql_connection_type: SqlDialect, **kwargs) -> 'SqlConnectionOptions':
        """
        Function signatures for factory method

        Postgres: (dialect: SqlDialects, host: str, port: int, username: str, password: str,
        database_name: str, timeout: int = None)
        """
        return SqlConnectionFactories.get_factory(sql_connection_type)(**kwargs)

    def __init__(self, dialect: SqlDialect, host: str, port: int, username: str, password: str, database_name: str
                 , timeout_s: int = None):
        self.dialect: SqlDialect = dialect
        self.host: str = host
        self.port: int = port
        self.username: str = username
        self.password: str = password
        self.database_name: str = database_name
        self.timeout: int = timeout_s
        self.connection_string: str = None

    @classmethod
    @abstractmethod
    def schema_validate_arguments(cls, schema: Schema, parameters: Dict) -> Dict:
        pass


class PostgresConnectionOptions(SqlConnectionOptions):
    _factory_schema: Schema = Schema(
        {
            'host': str,
            'port': int,
            'username': str,
            'password': str,
            'database_name': str
            # 'timeout': int
        },
        ignore_extra_keys=True
    )

    def __init__(self,
                 dialect: SqlDialect,
                 host: str,
                 port: int,
                 username: str,
                 password: str,
                 database_name: str,
                 timeout_s: int = None) -> None:
        super().__init__(dialect, host, port, username, password, database_name, timeout_s)
        self.connection_string = \
            f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database_name}"

    @classmethod
    def schema_validate_arguments(cls, schema: Schema, parameters: Dict) -> Dict:
        return schema.validate(parameters)

    @classmethod
    def factory(cls, **kwargs) -> 'PostgresConnectionOptions':
        parameters: Dict = cls.schema_validate_arguments(cls._factory_schema, kwargs)
        return cls(SqlDialect.postgres, parameters['host'], parameters['port']
                   , parameters['username'], parameters['password'], parameters['database_name']
                   , parameters.get('timeout'))


class SqlConnectionFactories:
    _factories: Dict[SqlDialect, Callable] = {
        SqlDialect.postgres: PostgresConnectionOptions.factory
        # , SqlDialects.sqlite: SqliteConnectionOptions.factory
    }

    @classmethod
    def get_factory(cls, factory_type: SqlDialect) -> Callable:
        return cls._factories[factory_type]


class SqlInterface:
    """SQL methods to tack onto SQL based librarians"""
    def __init__(self, connection_options: SqlConnectionOptions) -> None:
        self.connection_options = connection_options
        self.sql_engine: Engine = None
        self.sql_metadata: sqlalchemy.MetaData = None

    def update(self, schema: str, table: str, column: str, value: Any, sql_connection: Connection) -> None:
        raise NotImplementedError

    def select(self, schema: str, table: str, sql_connection: Connection) -> List[Dict[str, Any]]:
        sql_table: Table = self._get_table_reflection(schema, table)
        return self._execute_query(sql_connection, sql_table.select())

    def insert(self, schema: str, table: str, values: List[Dict[str, Any]], sql_connection: Connection) -> None:
        sql_table: Table = self._get_table_reflection(schema, table)
        insert_query = sql_table.insert(values=values)
        self._execute_query(sql_connection, insert_query)

    def setup_pre_connection(self, connection_options) -> None:
        self._build_engine(connection_options)
        self._metadata_reflection(self.sql_engine)

    def close_connection(self, sql_connection: Connection) -> None:
        if sql_connection is not None:
            sql_connection.close()

    @contextmanager
    def managed_connection(self, connection_options: SqlConnectionOptions = None) -> Connection:
        if connection_options is None:
            connection_options = self.connection_options

        self.setup_pre_connection(connection_options)
        connection: Connection = None
        try:
            connection = self.sql_engine.connect()
            yield connection
        finally:
            self.close_connection(connection)

    # SQLAlchemy internal methods
    def _build_engine(self, connection_options: SqlConnectionOptions) -> None:
        self.sql_engine = sqlalchemy.create_engine(connection_options.connection_string)

    def _metadata_reflection(self, sql_engine) -> None:
        self.sql_metadata = sqlalchemy.MetaData(bind=sql_engine)

    def _get_table_reflection(self, schema: str, table: str) -> Table:
        return Table(table, self.sql_metadata, schema=schema, autoload=True)

    def _validate_write_schema(self, table: Table, values: Dict[str, Any]) -> bool:
        table_columns = list(dict(table.columns).keys())
        return list(values.keys()) == table_columns

    def _parse_result_proxy(self, result) -> List[Dict[str, Any]]:
        return list(map(lambda x: dict(x), result))

    def _execute_query(self, sql_connection: Connection, sql_query: Query) -> List[Dict[str, Any]]:
        start_time: float = time.time()
        return_result: List[Dict[str, Any]] = None
        try:
            result: ResultProxy = sql_connection.execute(sql_query)
            if result.returns_rows:
                return_result: List[Dict[str, Any]] = self._parse_result_proxy(result)
        except Exception as e:
            logger.info(f"SQL query failed: {e}")
            logger.debug(f"SQL query {str(sql_query.compile())}, connection: {sql_connection.engine} failed with exception {e}")
            raise e
        finally:
            end_time: float = time.time()
            query_time: float = end_time - start_time
            logger.info(f"SQL execute time: {query_time}")
            logger.debug(
                f"SQL execute time: {query_time}, query: {str(sql_query.compile())}, connection: {sql_connection.engine}"
            )

        return return_result

