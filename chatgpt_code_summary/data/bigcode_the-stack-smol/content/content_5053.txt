import psycopg2


# Returns connection to the DB
def get_sql_connection():
    conn = psycopg2.connect(user="cqwhbabxmaxxqd",
                            password="a3063dc5aeec69b41564cd0f1e3c698e0ff9653385f3b87c0f113b70951eb5b3",
                            host="ec2-54-235-92-244.compute-1.amazonaws.com",
                            port="5432",
                            database="d8d34m4nml4iij")
    return conn
