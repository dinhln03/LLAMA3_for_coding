from config.config import db
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    username = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    email = db.Column(db.String)

    def __init__(self, username, email, password):
        self.username = username
        self.password = password
        self.email = email
