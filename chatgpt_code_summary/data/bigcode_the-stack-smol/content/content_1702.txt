from peewee import *
import psycopg2
import datetime

db = PostgresqlDatabase("prueba", host="localhost", port=5432, user="postgres", password="P@ssw0rd")

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    Username = CharField(unique = True)
    email = CharField(unique = True)
    created_date = DateTimeField(default= datetime.datetime.now)
    class Meta:
        db_table = 'Users'

if __name__== '__main__':
    if not User.table_exists():
        User.create_table()
    query_1 = User.select().where( User.Username == "Raul").get()
    print (query_1.email)
    for all_users in User.select():
        print (all_users.Username)
    
