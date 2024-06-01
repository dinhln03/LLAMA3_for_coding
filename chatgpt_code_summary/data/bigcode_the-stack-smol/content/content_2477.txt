import psycopg2
import psycopg2.extras


class DBHandler:

    """
    
    Handles I/O concerning the database to hide its implementation from client services.
    
    """

    def __init__(self,
                 postgres_username=None,
                 postgres_password=None,
                 db_username='dbpedia_app',
                 db_password='dummy_password'):

        # ordinarily you would get these from some secret store
        # e.g. heroku has a specific url that you parse to get both
        # or os.environ storage (like those used for API keys and the like)
        user_name = db_username
        password = db_password

        # check to see if the db exists locally, create it if necessary
        if postgres_password is not None and postgres_username is not None:

            try:
                connection = psycopg2.connect("dbname='postgres' user='%s' "
                                              "host='localhost' password='%s'"
                                              % (postgres_username, postgres_password))
                connection.autocommit = True
                cursor = connection.cursor()

                # queries the postgres catalog to see if 'dbpedia' exists
                # if not, creates it
                cursor.execute("SELECT COUNT(*) = 0 FROM pg_catalog.pg_database WHERE datname = 'dbpedia'")
                not_exists_row = cursor.fetchone()
                not_exists = not_exists_row[0]
                if not_exists:
                    cursor.execute("CREATE USER %s PASSWORD '%s'" % (user_name, password))
                    cursor.execute('CREATE DATABASE dbpedia OWNER %s' % (user_name,))

                connection.close()

            except:
                # Presume if credentials are passed the user wants to perform this check/DB construction
                # fail via error propagation
                raise

        try:
            self.connection = psycopg2.connect("dbname='dbpedia' user='%s' host='localhost' password='%s'"
                                          % (user_name, password))
        except:
            raise AssertionError('Failed to connect to dbpedia database. Has the local dbpedia been created?')

    def __del__(self):

        self.connection.close()

    def commit(self):

        self.connection.commit()

    def schema_exists(self):

        """
        
        Checks the estimated number of tuples in the subjects table to determine if data exists
        
        :return: 
        """

        with self.connection.cursor() as cursor:

            cursor.execute('select reltuples FROM pg_class where relname = %s', ('subjects',))
            result = cursor.fetchone()[0]
            return result > 0

    def build_table_schema(self, schema_name, schema_file_path):

        """
        
        Loads the dbpedia schema used for supporting downstream analysis. If the schema already exists, it is
        dropped (deleted) and recreated.
        
        :param schema_name: 
        :param schema_file_path: 
        :return: 
        """

        # do not call with user input given the manual query construction here

        with self.connection.cursor() as cursor:

            cursor.execute('DROP SCHEMA IF EXISTS %s CASCADE' % schema_name)
            schema_file = open(schema_file_path, 'rU').read()
            cursor.execute(schema_file)

    def build_indices(self):

        """
        
        Builds the following indices:
        
        Index on name for subjects
        Index on predicate for predicate_object
        Index on subject_id for predicate object
        
        :return: 
        """

        with self.connection.cursor() as cursor:

            cursor.execute('DROP INDEX IF EXISTS dbpedia.pv_subject_id_idx')
            cursor.execute('DROP INDEX IF EXISTS dbpedia.subject_idx')
            cursor.execute('DROP INDEX IF EXISTS dbpedia.pv_predicate_idx')

            cursor.execute('create index subject_idx on dbpedia.subjects (name)')
            cursor.execute('create index pv_subject_id_idx on dbpedia.predicate_object (subject_id)')
            cursor.execute('create index pv_predicate_idx on dbpedia.predicate_object (predicate);')

    def insert_spo_tuple(self, spo_tuple):

        """
        
        Handles the insertion of spo tuples into the db. Workflow:
        
        Attempt to find the subject table entry corresponding to your subject. If found, use that ID for
        inserting your po values. Otherwise, insert your subject into the subject table and use that ID
        instead. The resulting id, predicate, object tuple is then inserted into the predicate_object table.
        
        :param spo_tuple: 
        :return: 
        """

        (subject, predicate, db_object) = spo_tuple

        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:

            cursor.execute('select subject_id from dbpedia.subjects '
                           'where name = %s', (subject,))

            results = cursor.fetchone()

            if results is None or len(results) == 0:

                cursor.execute('INSERT INTO dbpedia.subjects (name) VALUES (%s) '
                               'returning subject_id', (subject,))
                results = cursor.fetchone()

            id = results['subject_id']

            # now we have the correct id in either case, insert the values into the db

            cursor.execute('INSERT INTO dbpedia.predicate_object (subject_id, predicate, object) '
                           'VALUES (%s, %s, %s)', (id, predicate, db_object))

    def get_person_metadata(self, person_name, use_exact_match=False):

        """
        
        Returns all metadata associated with the provided person_name. However, does not actually check
        to see if the identifier corresponds to a person or not; the class of the identifier will
        be included in the returned metadata though. DBPedia People only contains people predicate
        types as well.
        
        Use_exact_match toggles between two behaviors: if True, then uses the exact identifier provided
        to query against the subject table (WHERE = identifier). If False, uses the LIKE operator
        to attempt to find similar IDs that are not exactly the same. Results will still be a superset
        of the use_exact_match = True case.
        
        :param person_name: 
        :param use_exact_match:
        :return: 
        """

        # wikipedia replaces all spaces with under scores
        # upper case to make case sensitive
        person_name = person_name.replace(' ', '_').upper()

        with self.connection.cursor() as cursor:

            # get id associated with this person
            # get all similar IDs

            if not use_exact_match:
                cursor.execute('SELECT subject_id, name FROM dbpedia.subjects WHERE upper(name) '
                               'LIKE %s',
                               ('%%' + person_name + '%%',))
            else:
                cursor.execute('SELECT subject_id, name FROM dbpedia.subjects WHERE upper(name) = %s',
                               (person_name,))

            results = cursor.fetchall()

            # no person matches the input name
            # return empty list
            if results is None:
                return []

            subject_id_list = [x[0] for x in results]

            # get all metadata associated with the subject_ids
            cursor.execute('select dbpedia.subjects.name, predicate, object '
                           'FROM dbpedia.predicate_object '
                           'INNER JOIN dbpedia.subjects on (dbpedia.subjects.subject_id = dbpedia.predicate_object.subject_id) '
                           'WHERE dbpedia.predicate_object.subject_id = ANY(%s)', (subject_id_list,))

            # this should never be none
            # Sort results by name and return
            return sorted(cursor.fetchall(), key=lambda x: x[0])

    def get_tuples_by_predicate(self, predicate_of_interest):

        """
        
        Extracts SPO tuples based on the predicate value passed to the function. This query will be slow since
        you are querying such a large fraction of the po table at once (unless your predicate does not exist).
        
        Predicates:
        
        Name
        Type
        Gender
        Description
        Birthdate
        GivenName
        Surname
        BirthPlace
        DeathDate
        DeathPlace
        
        :param predicate_of_interest: 
        :return: 
        """

        with self.connection.cursor() as cursor:

            cursor.execute('select dbpedia.subjects.name, '
                           'predicate, '
                           'object '
                           'FROM dbpedia.predicate_object '
                           'INNER JOIN dbpedia.subjects on (dbpedia.subjects.subject_id = dbpedia.predicate_object.subject_id) '
                           'WHERE upper(dbpedia.predicate_object.predicate) = upper(%s)', (predicate_of_interest,))

            results = cursor.fetchall()

            if results is None:
                return []
            else:
                return results
