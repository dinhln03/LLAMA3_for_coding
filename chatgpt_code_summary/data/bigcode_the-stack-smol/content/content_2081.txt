import os
from access import Access
from user import User

log = os.path.dirname(os.path.abspath(__file__)) + "/temp/access.log"

class UserDAO(object):
    __database = None
    __cursor = None 
    def __init__(self):
        self.__database = Access()
        self.__cursor = self.__database.getCursor()
        self.initDatabase()
    def initDatabase(self):
        try:
            self.__cursor.execute(""" create table user (name text, username text, password text) """)
            self.__database.commit()
        except:
            pass
        
    def insert(self, user):
        if len(self.getUser(user.getUsername())) == 0:  
            users = [(user.getName(), user.getUsername() , user.getPassword()), ]
            self.__cursor.executemany("INSERT INTO user VALUES (?,?,?)", users)
            self.__database.commit()
    def update(self, user):
        users = [(user.getName(),user.getPassword(), user.getUsername())]
        self.__cursor.executemany("UPDATE user SET name = ?, password = ? where username = ? ", users)
        self.__database.commit()
    def delete(self, username):
        self.__cursor.execute("DELETE FROM user WHERE username = " + username)
        self.__database.commit()
    def list(self):
        self.__cursor.execute("SELECT * FROM user")
        print self.__cursor.fetchall()
    def getUser(self, username):
        self.__cursor.execute("SELECT * FROM user WHERE username = ?",[(username)] )
        return self.__cursor.fetchall()
    
    def log(self, user, request):
        flines = user.toString() + " >>> " + request + "\n"
        f = open(log, 'a')
        f.writelines([flines,])
        f.close()
        
        
        