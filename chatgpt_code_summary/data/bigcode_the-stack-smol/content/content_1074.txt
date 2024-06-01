import os

from databases import Database
from sqlalchemy import MetaData, create_engine

SQLALCHEMY_DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or '{}://{}:{}@{}:{}/{}'.format(
        os.environ.get("DATABASE"),
        os.environ.get("DB_USERNAME"),
        os.environ.get("DB_PASSWORD"),
        os.environ.get("DB_HOST"),
        os.environ.get("DB_PORT"),
        os.environ.get("DB_NAME"),
    )
)

database = Database(
    SQLALCHEMY_DATABASE_URL,
    ssl=False,
    min_size=5,
    max_size=20,
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    echo=False,
)

metadata = MetaData()
