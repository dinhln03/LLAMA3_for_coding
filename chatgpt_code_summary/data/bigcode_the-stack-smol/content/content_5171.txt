# Database Lib
"""
Oracle
PostGresSQL
SQLite
SQLServer
Hive
Spark
"""
import os, datetime, pandas, time, re
from collections import namedtuple, OrderedDict

import jmespath
import sqlalchemy
from multiprocessing import Queue, Process

from xutil.helpers import (
  log,
  elog,
  slog,
  get_exception_message,
  struct,
  now,
  get_databases,
  get_dir_path,
  get_profile,
  get_variables,
  file_exists,
  str_rmv_indent,
  ptable,
  make_rec,
  get_error_str,
)
from xutil.diskio import read_yaml, write_csvs

conns = {}

_fwklike = lambda k, v: "lower({}) like lower('{}')".format(k, v)
_fwkeq = lambda k, v: "{} = '{}'".format(k, v)
_fw = lambda sep, _fwkop, **kws: sep.join([_fwkop(k, v) for k, v in kws.items()])  # Format WHERE
fwa = lambda _fwkop=_fwkeq, **kws: _fw(' and ', _fwkop, **kws)  # Format WHERE AND
fwo = lambda _fwkop=_fwkeq, **kws: _fw(' or ', _fwkop, **kws)  # Format WHERE OR
rows_to_dicts = lambda rows: [row._asdict() for row in rows]


class DBConn(object):
  """Base class for database connections"""

  _fix_f_name = lambda self, f: f
  _to_text = lambda self, t: t

  def __init__(self, conn_dict, profile=None, echo=False):
    "Inititate connection"
    self._cred = struct(conn_dict)
    self._cred.kwargs = conn_dict.get('kwargs', {})
    self.name = self._cred.get('name', None)
    self.username = self._cred.get('username', None)
    self.type = self._cred.type
    self.engine = None
    self._cursor_description = None
    self.profile = profile
    self.batch_size = 10000
    self.fetch_size = 20000
    self.echo = echo
    self.connect()
    self.last_connect = now()

    # Base Template
    template_base_path = '{}/database/templates/base.yaml'.format(
      get_dir_path())
    self.template_dict = read_yaml(template_base_path)

    # Specific Type Template
    template_path = '{}/database/templates/{}.yaml'.format(
      get_dir_path(), self.type)
    temp_dict = read_yaml(template_path)

    for key1 in temp_dict:
      # Level 1
      if isinstance(temp_dict[key1], dict):
        if key1 not in self.template_dict:
          self.template_dict[key1] = temp_dict[key1]

        # Level 2
        for key2 in temp_dict[key1]:
          # Always Overwrite
          self.template_dict[key1][key2] = temp_dict[key1][key2]
      else:
        # Level 1 Non-Dict Overwrite
        self.template_dict[key1] = temp_dict[key1]

    self.variables = self._template('variables')

    if os.getenv('PROFILE_YAML'):
      other_vars = get_variables()
      for key in other_vars:
        self.variables[key] = other_vars[key]

    self.tmp_folder = self.variables['tmp_folder']
    self.set_variables()

    if echo:
      log("Connected to {} as {}".format(self._cred.name, self._cred.user))

  def connect(self):
    """Connect to Database"""
    self.engine = self.get_engine()
    self.connection = self.engine.connect()
  
  def close(self):
    """Close database connection"""
    self.conn.connection.close()

  def reconnect(self, min_tresh=0):
    """Re-Connect to Database if minute threshold reached"""
    if (now() - self.last_connect).total_seconds() > min_tresh * 60:
      log('Reconnecting to {}...'.format(self.name))
      self.connect()
      self.last_connect = now()

  def set_variables(self):
    """Set custom variables"""
    raise Exception("Method 'set_variables' is not implemented!")

  def get_dialect(self, echo=False):
    """SQLAlchemy dialect"""
    raise Exception("Method 'get_dialect' is not implemented!")

  def get_engine(self, echo=False):
    import sqlalchemy
    if not self.engine:
      self.create_engine(echo=self.echo)
    self.engine_inspect = sqlalchemy.inspect(self.engine)
    return self.engine

  def check_pk(self, table, fields):
    "Check Primary key to ensure there are not duplicates"
    if 'where' in fields.lower():
      fields, where_clause = fields.lower().split('where')
      where_clause = 'where ' + where_clause
    else:
      where_clause = ''
    sql = '''
    select
      '{table}' as table,
      case when count(1) = count({fields}) then 'PASS' else 'FAIL' end as pk_result
    from {table}
    {where_clause}
    '''.format(
      table=table,
      fields=fields,
      where_clause=where_clause,
    )
    data = self.query(sql, echo=False)
    headers = self._fields
    print(ptable(headers, data))
    if data[0].pk_result == 'FAIL':
      raise (Exception('PK Text failed for table "{}" with fields "{}"'.format(
        table, fields)))

  def _do_execute(self, sql):
    try:
      self._cursor_description = None
      self.fields = None
      self.result = self.connection.execute(sql)
      self._cursor_description = self.result._cursor_description()
      self._fields = self._get_cursor_fields()
    except Exception as E:
      if 'not open' in get_error_str(E):
        pass # error when Oracle doesn't have a cursor open
      else:
        log(Exception('Error for SQL:\n' + sql))
        raise E

  def execute_multi(self,
                    sql,
                    dtype='namedtuple',
                    limit=None,
                    echo=True,
                    query_name='Record',
                    log=log):
    """
    Execute multiple SQL statements separtated by ';'. Returns a generator.
    Example:
      for fields, rows in conn.execute(sql):
        print(fields)
        print(len(rows))
    """

    self.reconnect(min_tresh=10)

    data = None
    fields = None
    rows = []
    message_mapping = {
      'drop ': 'Dropping {}.',
      'truncate ': 'Truncating {}.',
      'select ': 'Selecting {}.',
      'create ': 'Creating {}.',
      'insert ': 'Inserting {}.',
      'alter ': 'Altering {}.',
      'update ': 'Updating {}.',
      'delete ': 'Deleting {}.',
      'exec ': 'Calling Procedure {}.',
      'grant ': 'Granting {}.',
    }

    sqls = sql.split(';')

    for sql in sqls:
      if not sql.strip(): continue

      sql_ = sql.strip().lower()

      for word, message in message_mapping.items():
        if sql_.startswith(word):
          if echo:
            log(
              message.format(' '.join(
                sql_.splitlines()[0].split()[1:3]).upper()))
          break

      # Call procedure with callproc
      if sql_.startswith('exec '):
        procedure = sql_[5:].split('(')[0]
        args = sql_[5:].split('(')[1][:-1].replace("'", '').split(',')
        args = [a.strip() for a in args]
        cursor.callproc(procedure, args)
        continue

      try:
        self._fields = []
        rows = self.query(
          sql,
          rec_name=query_name,
          dtype=dtype,
          limit=limit,
          echo=echo,
          log=log)
        fields = self._fields

        if '-- pk_test:' in sql.lower() and sql_.startswith('create'):
          sql_lines = sql_.splitlines()
          regexp = r'create\s+table\s+(\S*)[\sa-zA-Z\d]+ as'
          table = re.findall(regexp, sql_lines[0])[0]
          line = [
            l for l in sql_lines if l.strip().lower().startswith('-- pk_test:')
          ][0]
          fields = line.split(':')[-1]
          self.check_pk(table, fields)

      except Exception as E:
        message = get_exception_message().lower()

        if sql_.startswith(
            'drop ') and self.error_msg['table_not_exist'] in message:
          log("WARNING: Table already dropped.")
        else:
          raise E

      if not fields: fields = []

      yield fields, rows

  def execute(self,
              sql,
              dtype='tuple',
              limit=None,
              echo=True,
              query_name='Record',
              log=log):
    """Execute SQL, return last result"""
    self.reconnect(min_tresh=10)

    data = None
    fields = None
    rows = []
    message_mapping = {
      'drop ': 'Dropping {}.',
      'truncate ': 'Truncating {}.',
      'select ': 'Selecting {}.',
      'create ': 'Creating {}.',
      'insert ': 'Inserting {}.',
      'alter ': 'Altering {}.',
      'update ': 'Updating {}.',
      'delete ': 'Deleting {}.',
      'exec ': 'Calling Procedure {}.',
      'grant ': 'Granting {}.',
    }

    sql_ = sql.strip().lower()

    for word, message in message_mapping.items():
      if sql_.startswith(word):
        if echo:
          log(
            message.format(' '.join(
              sql_.splitlines()[0].split()[1:3]).upper()))
        break

    # Call procedure with callproc
    if sql_.startswith('exec '):
      procedure = sql_[5:].split('(')[0]
      args = sql_[5:].split('(')[1][:-1].replace("'", '').split(',')
      args = [a.strip() for a in args]
      connection = self.engine.raw_connection()
      try:
        cursor = connection.cursor()
        cursor.callproc(procedure, args)
        self._fields = self._get_cursor_fields(cursor_desc=cursor.description)
        rows = list(cursor.fetchall())
        cursor.close()
        connection.commit()
        return fields, rows
      finally:
        connection.close()

    try:
      self._fields = []
      rows = self.query(
        sql,
        rec_name=query_name,
        dtype=dtype,
        limit=limit,
        echo=echo,
        log=log)
      fields = self._fields

      if '-- pk_test:' in sql.lower() and sql_.startswith('create'):
        sql_lines = sql_.splitlines()
        regexp = r'create\s+table\s+(\S*)[\sa-zA-Z\d]+ as'
        table = re.findall(regexp, sql_lines[0])[0]
        line = [
          l for l in sql_lines if l.strip().lower().startswith('-- pk_test:')
        ][0]
        fields = line.split(':')[-1]
        self.check_pk(table, fields)

    except Exception as E:
      message = get_exception_message().lower()

      if sql_.startswith(
          'drop ') and self.error_msg['table_not_exist'] in message:
        log("WARNING: Table already dropped.")
      else:
        raise E

    if not fields: fields = []

    return fields, rows

  def insert(self, table, data, echo=False):
    """Insert records of namedtuple or dicts"""
    raise Exception('insert not implemented')

  def drop_table(self, table, log=log):
    "Drop table"

    try:
      sql = self._template('core.drop_table').format(table)
      self._do_execute(sql)
    except Exception as E:
      message = get_exception_message().lower()
      if self._template('error_filter.table_not_exist') in message:
        if self.echo:
          log('Table "{}" already dropped.'.format(table))
      else:
        raise E

  def create_table(self, table, field_types, drop=False, log=log):
    "Create table"

    if drop:
      self.drop_table(table, log=log)

    new_ftypes = OrderedDict()
    for f in field_types:
      ftype, max_len, dec_len = field_types[f]
      if dec_len:
        suff = '({},{})'.format(max_len, dec_len)
      elif max_len:
        suff = '({})'.format(max_len)
      else:
        suff = ''

      new_ftypes[f] = self._template('general_type_map')[ftype].replace(
        '()', suff)

    field_types_str = ', \n'.join([
      self._fix_f_name(field) + ' ' + new_ftypes[field] for field in new_ftypes
    ])

    sql = self._template('core.create_table').format(
      table=table,
      col_types=field_types_str,
    )

    # log('Creating table: \n' + sql))
    try:
      self._do_execute(sql)
    except Exception as e:
      raise e

    log('Created table "{}"'.format(table))

  def _get_cursor_fields(self, as_dict=False, native_type=True, cursor_desc=None):
    "Get fields of active Select cursor"
    fields = OrderedDict()
    cursor_desc = cursor_desc if cursor_desc else self._cursor_description
    if cursor_desc == None:
      return []

    for f in cursor_desc:
      f_name = f[0].lower()
      if as_dict:
        if native_type:
          f_type = f[1]
        else:
          f_type = self.reverse_data_map[f[1]]

          # assign floa/double as needed
          if 'cx_Oracle.NUMBER' in str(f[1]):
            if f[4] and f[4] > 11: f_type = 'long'
            if f[5] and f[5] > 0: f_type = 'double'

        fields[f_name] = f_type
      else:
        fields[f_name] = None

    if as_dict:
      return fields
    else:
      return list(fields.keys())

  def stream(self,
             sql,
             rec_name='Record',
             dtype='namedtuple',
             yield_chuncks=False,
             chunk_size=None,
             limit=None,
             echo=True):
    "Stream Select from SQL, yield records as they come in"
    self.reconnect(min_tresh=10)
    if echo: log("Streaming SQL for '{}'.".format(rec_name))

    fetch_size = limit if limit else self.fetch_size
    fetch_size = chunk_size if chunk_size else fetch_size

    try:
      self._do_execute(sql)
    except Exception as e:
      raise e

    if dtype == 'tuple':
      make_rec = lambda row: row
      make_batch = lambda rows: rows
    elif dtype == 'dataframe':
      yield_chuncks=True
      make_batch = lambda rows: pandas.DataFrame(rows, columns=self._fields)
    else:
      Record = namedtuple(
        rec_name.replace(' ', '_').replace('.', '_'), self._fields)
      make_rec = lambda row: Record(*row)
      make_batch = lambda rows: [make_rec(r) for r in rows]

    self._stream_counter = 0

    while True:
      if not self._fields:
        break
      rows = self.result.fetchmany(fetch_size)
      if rows:
        if yield_chuncks:
          batch = make_batch(rows)
          self._stream_counter += len(batch)
          if len(batch):
            yield batch
        else:
          for row in rows:
            self._stream_counter += 1
            yield make_rec(row)
      else:
        break
      if limit:
        break
        
        

    # log('Stream finished at {} records.'.format(self._stream_counter))

  def query(self,
             sql,
             rec_name='Record',
             dtype='namedtuple',
             limit=None,
             echo=True,
             retrying=False,
             log=log):
    "Select from SQL, return list of namedtuples"
    # if echo: log("Running SQL for '{}'.".format(rec_name))

    self.reconnect(min_tresh=10)
    s_t = datetime.datetime.now()

    _data = list(self.stream(sql, dtype=dtype, echo=False, limit=limit))
    if not self.result.closed:
      self.result.close()

    fields = self._fields
    if not fields: return []

    if dtype == 'namedtuple':
      Record = namedtuple(rec_name.replace(' ', '_').replace('.', '_'), fields)
      if limit:
        data = [Record(*row) for row in _data]
      else:
        data = [Record(*row) for row in _data]

    elif dtype == 'tuple':
      if limit:
        data = [tuple(row) for row in _data]
      else:
        data = [tuple(row) for row in _data]

    elif dtype == 'dataframe':
      if limit:
        data = pandas.DataFrame([row for row in _data], columns=fields)
      else:
        data = pandas.DataFrame([row for row in _data], columns=fields)
    else:
      raise (Exception('{} is not recongnized.'.format(dtype)))

    secs = (datetime.datetime.now() - s_t).total_seconds()
    rate = round(len(data) / secs, 1)
    if echo:
      log(" >>> Got {} rows in {} secs [{} r/s].".format(
        len(data), secs, rate))
    return data

  def _split_schema_table(self, table_name):
    schema, table = table_name.split('.') if '.' in table_name else (
      self.username, table_name)
    return schema, table

  def _concat_fields(self, fields, as_text=False):
    return ' || '.join(fields)

  def _template(self, template_key_str):
    val = jmespath.search(template_key_str, self.template_dict)
    if isinstance(val, str):
      val = str_rmv_indent(val)
    return val

  def get_schemas(self, echo=True):
    "Get list of schemas."
    Rec = namedtuple('Schemas', 'schema')
    self._fields = Rec._fields

    sql_tmpl = self._template('metadata.schemas')
    if sql_tmpl:
      schemas = [r[0] for r in self.query(sql_tmpl)]
    else:
      # http://docs.sqlalchemy.org/en/rel_0_9/core/reflection.html#sqlalchemy.engine.reflection.Inspector.get_schemas
      self.get_engine(echo=echo)
      schemas = self.engine_inspect.get_schema_names()

    rows = [Rec(s) for s in schemas]
    return rows

  def get_objects(self, schema, object_type='all', echo=True):
    "Get metadata for objects. object_type in 'all', 'table', 'view'"
    Rec = namedtuple('Table', 'schema object_name object_type')
    self._fields = Rec._fields

    def get_rec(object_name, object_type):
      r_dict = dict(
        schema=schema, object_name=object_name, object_type=object_type)
      return Rec(**r_dict)

    if object_type == 'all':
      table_rows = self.get_tables(schema)
      rows = [get_rec(r.table, 'table') for r in sorted(table_rows)]
      view_rows = self.get_views(schema)
      rows += [get_rec(r.view, 'view') for r in sorted(view_rows)]
    elif object_type == 'table':
      table_rows = self.get_tables(schema)
      rows = [get_rec(r.table, 'table') for r in sorted(table_rows)]
    elif object_type == 'view':
      view_rows = self.get_views(schema)
      rows += [get_rec(r.view, 'view') for r in sorted(view_rows)]
    else:
      raise Exception('Object type "{}" not supported!'.format(object_type))

    return rows

  def get_tables(self, schema, echo=True):
    "Get metadata for tables."
    schemas = schema if isinstance(schema, list) else [schema]

    def get_tables_for(schema):
      def get_rec(table):
        self._fields = ['schema', 'table']
        return tuple([schema, table])

        # Getting pickle.PicklingError: Can't pickle <class 'xutil.database.base.Table'>
        Rec = namedtuple('Table', 'schema table')
        self._fields = Rec._fields
        r_dict = dict(schema=schema, table=table)
        return Rec(**r_dict)

      sql_tmpl = self._template('metadata.tables')
      if sql_tmpl:
        tables = self.query(sql_tmpl.format(schema=schema))
        if hasattr(self, '_std_get_tables'):
          tables = self._std_get_tables(schema, tables)
      else:
        self.get_engine(echo=echo)
        tables = self.engine_inspect.get_table_names(schema)
      
      return [get_rec(v) for v in sorted(tables)]
    
    rows = []
    for schema in schemas:
      for row in get_tables_for(schema):
        rows.append(row)

    return rows

  def get_views(self, schema, echo=True):
    "Get metadata for views."
    schemas = schema if isinstance(schema, list) else [schema]

    def get_views_for(schema):
      def get_rec(view):
        self._fields = ['schema', 'view']
        return tuple([schema, view])

        # pickle.PicklingError: Can't pickle <class 'xutil.database.base.View'>
        Rec = namedtuple('View', 'schema view')
        self._fields = Rec._fields
        r_dict = dict(schema=schema, view=view)
        return Rec(**r_dict)

      sql_tmpl = self._template('metadata.views')
      if sql_tmpl:
        views = [r[0] for r in self.query(sql_tmpl.format(schema=schema))]
      else:
        self.get_engine(echo=echo)
        views = self.engine_inspect.get_view_names(schema)
      
      return [get_rec(v) for v in sorted(views)]

    rows = []
    for schema in schemas:
      for row in get_views_for(schema):
        rows.append(row)
    
    return rows

  def get_columns(self,
                  table_name,
                  object_type=None,
                  echo=False,
                  include_schema_table=True,
                  native_type=True):
    "Get column metadata for table"
    if include_schema_table:
      headers = 'schema table id column_name type nullable default autoincrement'
    else:
      headers = 'id column_name type  nullable default autoincrement'

    Rec = namedtuple('Columns', headers)
    self._fields = Rec._fields
    all_rows = []

    table_names = table_name if isinstance(table_name, list) else [table_name]

    for table_name in table_names:
      schema, table = self._split_schema_table(table_name)

      def get_rec(r_dict, column_order):
        if include_schema_table:
          r_dict['schema'] = schema
          r_dict['table'] = table
        r_dict['column_name'] = r_dict['name']
        r_dict['type'] = str(r_dict['type'])
        if not native_type:
          r_dict['type']= r_dict['type'].lower()
          r_dict['type'] = r_dict['type'].split('(')[0] if '(' in r_dict[
            'type'] else r_dict['type']
          native_type_map = self._template('native_type_map')
          if not r_dict['type'] in native_type_map:
            raise Exception('Field type "{}" not in native_type_map for {}'.format(r_dict['type'], self.type))
          r_dict['type'] = native_type_map[r_dict['type']]
        r_dict['id'] = column_order

        for k in list(r_dict):
          if k not in headers.split():
            del r_dict[k]

        if '(' in r_dict['type']:
          r_dict['type'] = r_dict['type'].split('(')[0]

        return Rec(**r_dict)

      sql_tmpl = self._template('metadata.columns')
      if sql_tmpl:
        rows = self.query(sql_tmpl.format(table=table, schema=schema))
        if hasattr(self, '_std_get_columns'):
          rows = self._std_get_columns(schema, table, rows)
      else:
        self.get_engine(echo=echo)
        rows = self.engine_inspect.get_columns(table, schema=schema)

      all_rows += [get_rec(r_dict, i + 1) for i, r_dict in enumerate(rows)]

    self._fields = Rec._fields
    return all_rows

  def get_primary_keys(self, table_name, echo=False):
    "Get PK metadata for table"
    Rec = namedtuple('PKs', 'schema table pk_name column_name column_order')
    self._fields = Rec._fields
    schema, table = self._split_schema_table(table_name)

    def get_rec(col, pk_name, column_order):
      r_dict = {}
      r_dict['schema'] = schema
      r_dict['table'] = table
      r_dict['pk_name'] = pk_name
      r_dict['column_name'] = col
      r_dict['column_order'] = column_order
      return Rec(**r_dict)

    sql_tmpl = self._template('metadata.primary_keys')
    if sql_tmpl:
      rows = self.query(sql_tmpl.format(table=table, schema=schema))
    else:
      self.get_engine(echo=echo)
      r_dict = self.engine_inspect.get_pk_constraint(table, schema=schema)
      rows = [
        get_rec(col, r_dict['name'], i + 1)
        for i, col in enumerate(r_dict['constrained_columns'])
      ]

    return rows

  def get_indexes(self, table_name, echo=False):
    "Get indexes metadata for table"
    Rec = namedtuple(
      'Indexes', 'schema table index_name column_name column_order unique')
    self._fields = Rec._fields
    schema, table = self._split_schema_table(table_name)

    def get_rec(r_dict):
      r_dict['schema'] = schema
      r_dict['table'] = table
      r_dict['index_name'] = r_dict['name']
      r_dict['unique'] = str(r_dict['unique'])
      del r_dict['name']
      for i, col in enumerate(r_dict['column_names']):
        r_dict['column_name'] = col
        r_dict['column_order'] = i + 1
        yield Rec(**r_dict)

    sql_tmpl = self._template('metadata.indexes')
    if sql_tmpl:
      rows = self.query(sql_tmpl.format(table=table, schema=schema))
    else:
      self.get_engine(echo=echo)
      rows = self.engine_inspect.get_indexes(table, schema=schema)
      rows = [get_rec(r_dict) for r_dict in rows]

    return rows

  def get_ddl(self, table_name, object_type=None, echo=True):
    "Get ddl for table"
    Rec = namedtuple('DDL', 'ddl')
    self._fields = Rec._fields
    schema, table = self._split_schema_table(table_name)

    sql_tmpl = self._template('metadata.ddl')
    if sql_tmpl:
      rows = self.query(
        sql_tmpl.format(
          schema=schema,
          table=table,
          obj_type=object_type,
        ))
    else:
      self.get_engine(echo=echo)
      ddl = self.engine_inspect.get_view_definition(table, schema=schema)
      rows = [Rec(ddl)] if ddl else []

    self._fields = Rec._fields
    return rows

  def get_all_columns(self):
    "Get all columns for all tables / views"
    sql_tmpl = self._template('metadata.all_columns')
    if not sql_tmpl:
      raise Exception('get_all_columns not implemented for {}'.format(
        self.type))

    rows = self.query(sql_tmpl)
    return rows

  def get_all_tables(self, filter, as_sql=False):
    "Get all tables / views"
    sql_tmpl = self._template('metadata.all_tables')
    if not sql_tmpl:
      raise Exception('get_all_tables not implemented for {}'.format(self.type))

    sql = sql_tmpl.format(filter=filter)
    return sql if as_sql else self.query(sql, echo=False)

  def analyze_fields(self,
                     analysis,
                     table_name,
                     fields=[],
                     as_sql=False,
                     union=True,
                     expr_func_map={},
                     **kwargs):
    """Base function for field level analysis
      expr_func_map: contains mapping for expression to SQL function to all fields
    """
    if '.' not in table_name:
      raise Exception("table_name must have schema and name in it with a '.'")
    if analysis not in self.template_dict['analysis']:
      raise Exception("'{}' not found in template for '{}'.".format(
        analysis, self.type))

    schema, table = self._split_schema_table(table_name)

    # get field type
    field_rows = self.get_columns(table_name)
    field_type = {r.column_name.lower(): r.type for r in field_rows}

    if not fields:
      fields = [r.column_name for r in field_rows]

    for expr in list(expr_func_map):
      tmpl_path = 'function.' + expr_func_map[expr]
      expr_func_map[expr] = ',\n'.join([
        self._template(tmpl_path).format(field=field)
        for field in [r.column_name for r in field_rows]
      ])

    sep = ' \nunion all\n' if union else ' \n ;\n'
    sql = sep.join([
      self._template('analysis.' + analysis).format(
        schema=schema,
        field=field,
        table=table,
        type=field_type[field.lower()] if field else '',
        **expr_func_map,
        **kwargs) for field in fields
    ])
    return sql if as_sql else self.query(sql, analysis, echo=False)

  def analyze_tables(self, analysis, tables=[], as_sql=False, **kwargs):
    """Base function for table level analysis"""
    if analysis not in self.template_dict['analysis']:
      raise Exception("'{}' not found in template for '{}'.".format(
        analysis, self.type))

    if not tables and 'schema' in kwargs:
      # get all tables
      rows = self.get_schemas(kwargs['schema'])
      crt_obj = lambda r: struct(dict(schema=r.schema, table=r.object_name))
      objs = [crt_obj(r) for r in rows]
    else:
      crt_obj = lambda schema, table: struct(dict(schema=schema, table=table))
      objs = [crt_obj(*self._split_schema_table(t)) for t in tables]

    sql = ' \nunion all\n'.join([
      self._template('analysis.' + analysis).format(
        schema=obj.schema, table=obj.table, **kwargs) for obj in objs
    ])

    return sql if as_sql else self.query(sql, analysis, echo=False)


  def analyze_join_match(self,
                        t1,
                        t2,
                        t1_field,
                        t2_field,
                        t1_filter='1=1',
                        t2_filter='1=1',
                        as_sql=False,
                        as_text=True,
                        lowercase=True):
    def get_kwargs(t1, t2, t1_field, t2_field, t1_filter, t2_filter):
      t1_field_arr = ['t1.' + f for f in t1_field.split(',')]
      t2_field_arr = ['t2.' + f for f in t2_field.split(',')]
      t1_field_concat = self._concat_fields(t1_field_arr, as_text=as_text)
      t2_field_concat = self._concat_fields(t2_field_arr, as_text=as_text)
      to_text = self._to_text

      if lowercase:
        conds = ' and '.join([
          'lower({}) = lower({})'.format(to_text(f), to_text(t2_field_arr[i]))
          for i, f in enumerate(t1_field_arr)
        ])
      else:
        conds = ' and '.join([
          '{} = {}'.format(to_text(f), to_text(t2_field_arr[i]))
          for i, f in enumerate(t1_field_arr)
        ])
      t1_fields1 = t1_field
      t2_fields1 = t2_field
      t1_field = ', '.join(['t1.' + f for f in t1_field_arr])
      t2_field = ', '.join(['t2.' + f for f in t2_field_arr])

      return dict(
        t1=t1,
        t1_field=t1_field_concat,
        t1_fields1=t1_fields1,
        t1_filter=t1_filter,
        t2=t2,
        t2_field=t2_field_concat,
        t2_fields1=t2_fields1,
        t2_filter=t2_filter,
        conds=conds,
      )

    kwargs = get_kwargs(
      t1=t1,
      t2=t2,
      t1_field=t1_field,
      t2_field=t2_field,
      t1_filter=t1_filter,
      t2_filter=t2_filter,
    )
    sql = self.analyze_fields(
      'table_join_match', t1, [''], as_sql=True, **kwargs)

    return sql if as_sql else self.query(sql, 'table_join_match', echo=False)


def get_conn(db,
             dbs=None,
             echo=True,
             reconnect=False,
             use_jdbc=False,
             conn_expire_min=10,
             spark_hive=False) -> DBConn:
  global conns

  dbs = dbs if dbs else get_databases()
  profile = get_profile()
  db_dict = struct(dbs[db])

  if db_dict.type.lower() == 'hive' and spark_hive:
    db_dict.type = 'spark'

  use_jdbc = True if (use_jdbc or ('use_jdbc' in db_dict
                                   and db_dict['use_jdbc'])) else use_jdbc

  if db in conns and not reconnect:
    if (now() - conns[db].last_connect).total_seconds() / 60 < conn_expire_min:
      return conns[db]

  if use_jdbc:
    log('*USING JDBC for ' + db)
    from .jdbc import JdbcConn
    conn = JdbcConn(db_dict, profile=profile)

  elif db_dict.type.lower() == 'oracle':
    from .oracle import OracleConn
    conn = OracleConn(db_dict, echo=echo)

  elif db_dict.type.lower() == 'spark':
    from .spark import SparkConn
    conn = SparkConn(db_dict, echo=echo)

  elif db_dict.type.lower() == 'hive':
    from .hive import HiveConn, Beeline
    if 'use_beeline' in db_dict and db_dict.use_beeline:
      conn = Beeline(db_dict, echo=echo)
    else:
      conn = HiveConn(db_dict, echo=echo)

  elif db_dict.type.lower() in ('postgresql', 'redshift'):
    from .postgresql import PostgreSQLConn
    conn = PostgreSQLConn(db_dict, echo=echo)

  elif db_dict.type.lower() == 'sqlserver':
    from .sqlserver import SQLServerConn
    conn = SQLServerConn(db_dict, echo=echo)

  elif db_dict.type.lower() == 'sqlite':
    from .sqlite import SQLiteConn
    conn = SQLiteConn(db_dict, echo=echo)
  else:
    raise Exception(f'Type {db_dict.type} not handled!')

  conns[db] = conn
  return conn


class SqlX:
  """
  SQL Express functions. Supports CRUD transactional operations.

  Suppose there is a table named 'cache', sqlx allows:
  
  sqlx.x('cache').insert(rows)
  sqlx.x('cache').insert_one(row)
  sqlx.x('cache').add(**kws)
  sqlx.x('cache').delete(where)
  sqlx.x('cache').update(rows, pk_fields)
  sqlx.x('cache').update_one(row, pk_cols)
  sqlx.x('cache').replace(rows, pk_fields)
  sqlx.x('cache').query(where)
  sqlx.x('cache').select_one(where)
  """

  def __init__(self, conn: DBConn, table, schema, ntRec: namedtuple):
    self.conn = conn
    self.table = table
    self.schema = schema
    self.ntRec = ntRec
    self.pk_fields = None
    self.table_obj = schema + '.' + table if schema else table

    self.insert_one = lambda row: self.insert([row])
    self.add = lambda **kws: self.insert([self.ntRec(**kws)])
    self.update_one = lambda row, pk_cols=None: self.update([row], pk_cols)
    self.update_rec=lambda pk_cols=None, **kws: self.update([make_rec(**kws)], pk_cols)
    self.replace_one = lambda row, pk_cols=None: self.replace([row], pk_cols)
    self.replace_rec=lambda pk_cols=None, **kws: self.replace([make_rec(**kws)], pk_cols)
    # self.select_one = lambda where: self.select_one(where, one=True)

  def _get_pk(self):
    if not self.pk_fields:
      pk_rows = self.conn.get_primary_keys(self.table_obj)
      self.pk_fields = [r.column_name for r in pk_rows]
    return self.pk_fields

  def insert(self, data):
    return self.conn.insert(self.table_obj, data)

  def update(self, data, pk_fields=None):
    if not pk_fields:
      pk_fields = self._get_pk()
      if not pk_fields:
        raise Exception("Need Keys to perform UPDATE!")
      t_fields = [x.lower() for x in data[0]._fields]
      for f in pk_fields:
        if not f.lower() in t_fields:
          # if keys not provided, need to make sure PK values are provided in data records
          raise Exception(
            "Value of  PK field '{}' must be provided to perform UPDATE!".
            format(f))

    self.conn.update(self.table_obj, data, pk_fields, echo=False)

  def update_one(self, row, pk_cols=None):
    self.update([row], pk_cols)

  def update_rec(self, pk_cols=None, **kws):
    self.update([make_rec(**kws)], pk_cols)

  def replace(self, data, pk_fields=None):
    if not pk_fields:
      pk_fields = self._get_pk()
    self.conn.replace(self.table_obj, data, pk_fields, echo=False)

  # def replace_rec(self, pk_cols=None, **kws):
  #   # add default None?
  #   for field in self.ntRec._fields:
  #     kws[field] = kws.get(field, None)

  #   self.replace([self.ntRec(**kws)], pk_cols)

  def query(self, where='1=1', one=False, limit=None, as_dict=False):
    rows = self.conn.query(
      "select * from {} where {}".format(self.table_obj, where),
      echo=False,
      limit=limit)
    rows = rows_to_dicts(rows) if as_dict else rows
    if one: return rows[0] if rows else None
    else: return rows

  def select_one(self, where, field=None, as_dict=False):
    row = self.query(where, one=True, as_dict=as_dict)
    if field and row:
      return row[field] if as_dict else row.__getattribute__(field)
    return row

  def delete(self, where):
    self.conn.execute("delete from {} where {}".format(self.table_obj, where))


def make_sqlx(conn, schema, tables):
  "Make sqlx lookup function for given tables"

  table_func_map = {}

  for table in tables:
    ntRec = namedtuple(table, tables[table].columns.keys())
    table_func_map[table] = SqlX(conn, table, schema, ntRec)

  # return table_func_map

  def sqlx(expr) -> SqlX:
    obj = jmespath.search(expr, table_func_map)
    if not obj:
      raise Exception('sqlx: Cannot find "{}"'.format(expr))
    return obj

  return sqlx


def get_sql_sources(sql_text, echo=False):
  """Obtain the source tables of a query
  """

  import sqlparse

  # replace "as(" to "as (" # this trips up the sql parser in CTEs
  sql_text = re.sub(r"as\(", "as (", sql_text, 0, re.MULTILINE | re.IGNORECASE)

  statements = sqlparse.parse(sql_text)
  cte_aliases = set()
  sql_sources = {}

  def get_sources(statement):
    sources_dict = {}
    last_kw_from = False
    last_kw_join = False
    cte_mode = False
    last_tok = None
    done = False

    while not done:
      for tok in statement.tokens:

        if tok.is_group:
          if cte_mode and isinstance(tok, sqlparse.sql.IdentifierList):
            for tok2 in tok.tokens:
              if isinstance(tok2, sqlparse.sql.Identifier):
                for tok3 in tok2.tokens:
                  if isinstance(tok3, sqlparse.sql.Parenthesis):
                    cte_aliases.add(tok3.parent.normalized.lower())
                    sources_dict2 = get_sources(tok3)
                    sources_dict = {**sources_dict, **sources_dict2}
          elif isinstance(tok, sqlparse.sql.Parenthesis):
            sources_dict2 = get_sources(tok)
            sources_dict = {**sources_dict, **sources_dict2}
          else:
            for tok2 in tok.tokens:
              if isinstance(tok2, sqlparse.sql.Parenthesis):
                cte_aliases.add(tok2.parent.normalized.lower())
                sources_dict2 = get_sources(tok2)
                sources_dict = {**sources_dict, **sources_dict2}


        if (last_kw_from or last_kw_join) and last_tok.is_whitespace:
          if isinstance(tok, sqlparse.sql.IdentifierList):
            for tok2 in tok.tokens:
              if isinstance(tok2, sqlparse.sql.Identifier) and '(' in tok2.value:
                sources_dict2 = get_sources(tok2)
                sources_dict = {**sources_dict, **sources_dict2}
              elif isinstance(tok2, sqlparse.sql.Identifier) and tok2.normalized.lower() not in cte_aliases:
                if echo: log('+Table = ' + tok2.normalized.lower())
                sources_dict[tok2.normalized.lower()] = tok.parent

          elif isinstance(tok, sqlparse.sql.Identifier) and tok.normalized.lower() not in cte_aliases:
            if echo: log('+Table = ' + tok.normalized.lower())
            sources_dict[tok.normalized.lower()] = tok.parent


          last_kw_join = False

        if tok.is_keyword and tok.normalized == 'WITH':
          cte_mode = True
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'GROUP':
          last_kw_join = False
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'WHERE':
          last_kw_join = False
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'ORDER':
          last_kw_join = False
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'CREATE':
          cte_mode = True
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'SELECT':
          cte_mode = False
          last_kw_from = False
        elif tok.is_keyword and tok.normalized == 'FROM':
          last_kw_from = True
        elif tok.is_keyword and 'JOIN' in tok.normalized:
          last_kw_join = True

        last_tok = tok
      done = True
    return sources_dict

  for s, statement in enumerate(statements):
    has_from = False
    last_kw_create = False
    last_kw_create_table = False
    create_table = None

    for tok in statement.tokens:
      if isinstance(tok, sqlparse.sql.Identifier) and last_kw_create_table:
        create_table = tok.normalized
        last_kw_create_table = False
        last_kw_create = False
        if echo: log('-CREATE TABLE ' + create_table)
      if tok.is_keyword and tok.normalized == 'TABLE' and last_kw_create:
        last_kw_create_table = True
      if tok.is_keyword and tok.normalized == 'CREATE':
        last_kw_create = True
      if tok.is_keyword and tok.normalized == 'FROM':
        has_from = True
      last_tok = tok

    if has_from:
      sources_dict = get_sources(statement)
      if create_table:
        sql_sources[create_table] = sorted(sources_dict)
      else:
        sql_sources[s] = sorted(sources_dict)

  return sql_sources
