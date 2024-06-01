from pymongo import MongoClient
import json
from newsapi.database import mongo


class UserModel:

    def __init__(self, _id, username, password):
        self.id = _id
        self.username = username
        self.password = password

    @classmethod
    def find_by_username(cls, username):
        result = mongo.db.user.find_one({'username': username}) 
        if result:
            user = cls(_id=result['_id'], username=result['username'], password=result['password'])
        else:
            user = None

        return user

    @classmethod
    def find_by_id(cls, _id):
        result = mongo.db.user.find_one({'_id': _id}) 
        if result:
            user = cls(_id=result['_id'], username=result['username'], password=result['password'])
        else:
            user = None

        return user
