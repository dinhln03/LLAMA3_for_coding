import sqlite3

from checktheplug.models.Server import Server


"""
    Operations to manage accessing the server database.
"""
class ServerDao:
    """
        Sets up the object with the sql connection.
    """
    def __init__(self, settings):
        self.conn = sqlite3.connect(settings.database)
        
    """
        Add Server to the database.
    """
    def add(self, new_server):
        if new_server:
            try:
                with self.conn:
                    cur = self.conn.cursor()
                    cur.execute("INSERT INTO servers(host, url) values(?, ?)", (new_server.host, new_server.url))
                    return(Server(cur.lastrowid, new_server.host, new_server.url), None)
            except sqlite3.IntegrityError as er:
                return (None, "There was a db issue: " + str(er))
        else:
            return (None, "No server passed in")
        
    """
        Find all the servers for a particular app.
    """
    def find_by_app_id(self, app_id):
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute("SELECT id, host, url from servers where app_id = ?", (app_id,))
                server_rows = cur.fetchall()
                return (list(map(lambda x: Server(x[0], x[1], x[2], app_id), server_rows)), None)
        except Exception as er:
            return (None, "There was a db issue: " + str(er))
    
    """
        Find x number of available servers or all that are available.
    """
    def find_available_servers(self, quantity):
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute("SELECT id, host, url from servers where app_id = null limit = ?", (quantity,))
                server_rows = cur.fetchall()
                return (list(map(lambda x: Server(x[0], x[1], x[2], None), server_rows)), None)
        except Exception as er:
            return (None, "There was a db issue: " + str(er))

    """
        Retrieve all servers.
    """
    def retrieve_all_servers(self):
        try:
            with self.conn:
                cur = self.conn.cursor()
                cur.execute("SELECT id, host, url from servers")
                server_rows = cur.fetchall()
                return (list(map(lambda x: Server(x[0], x[1], x[2], None), server_rows)), None)
        except Exception as er:
            return (None, "There was a db issue: " + str(er))
        
    """
        Tie an app to a number of servers.
    """
    def tie_app_to_servers(self, app_id, available_servers):
        try:
            with self.conn:
                cur = self.conn.cursor()
                server_id_string = ', '.join("?" * available_servers)
                cur.execute("update servers set app_id = ? where id in ({0})".format(server_id_string), tuple([app_id] + available_servers))
                return (None, "ok")
        except Exception as er:
            return (None, "There was a db issue: " + str(er))