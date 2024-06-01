import pymongo


def connect_to_mongo(username="", password="", host="localhost", port=27017):
    credentials = ""
    if username and password:
        credentials = f"{username}:{password}@"
    connection_url = f"mongodb://{credentials}{host}:{port}"
    return pymongo.MongoClient(connection_url)
