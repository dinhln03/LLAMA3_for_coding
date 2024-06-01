from scrapy.utils.project import get_project_settings
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(get_project_settings().get("CONNECTION_STRING"))


def create_table(engine):
    """ create tables"""
    Base.metadata.create_all(engine)


class Parliament(Base):
    """Sqlalchemy deals model"""

    __tablename__ = "parliament"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column("name", String)
    date_born = Column("date_born", String)
    place_born = Column("place_born", String, nullable=True)
    profession = Column("profession", String, nullable=True)
    lang = Column("lang", String, nullable=True)
    party = Column("party", String, nullable=True)
    email = Column("email", String, nullable=True)
    url = Column("url", String, nullable=True)
    education = Column("education", String, nullable=True)
    pp = Column("pp", String)
    dob = Column("dob", String)
