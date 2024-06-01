"""This module contains logic for refreshing materialized views.

Materialized views don't get refreshed automatically after a bucardo initial
sync.  This module detects them and refreshes them.

Classes exported:
MatViews: Identify materialized views and refresh them on the secondary database.
"""
import psycopg2
from psycopg2 import sql

from plugins import Plugin


class MatViews(Plugin):
    """Identify materialized views and refresh them on the secondary database.

    Materialized views are identified based on the namespaces specified in the
    config.

    Methods exported:
    refresh: find and refresh materialized views
    """

    def __init__(self, cfg):
        """Create configuration settings that may not already be set.

        The user can either define the relevant namespaces specifically for the
        mat_views plugin, or the mat_views plugin can draw on the settings in the
        bucardo section of the config.  If neither exists, the script will throw an
        error.

        Keyword arguments:
        cfg: contents of the config file as a dictionary
        """
        super(MatViews, self).__init__(cfg)

        # Override or inherit certain params from the parent, depending on the config.
        self._set_inheritable_params('mat_views')

    def refresh(self):
        """Refresh materialized views.

        First, this method finds the namespaces being replicated, by referring to the
        config for schemas and tables.

        Then it finds any materialized views in the namespaces.

        Then it refreshes the materialized views.
        """
        print('Finding materialized views.')
        # 'm' is for "materialized view".
        views = self._find_objects('m', self.repl_objects)

        if views:
            conn = psycopg2.connect(self.secondary_schema_owner_conn_pg_format)
            for view in views:
                print(f'Refreshing {view[0]}.{view[1]}')
                query = sql.SQL('REFRESH MATERIALIZED VIEW {schema}.{table}').format(
                    schema=sql.Identifier(view[0]),
                    table=sql.Identifier(view[1])
                )
                try:
                    with conn.cursor() as cur:
                        cur.execute(query)
                        conn.commit()
                except Exception:
                    conn.close()
                    raise
            conn.close()
            print('Done refreshing views.')
        else:
            print('No materialized views found.')
