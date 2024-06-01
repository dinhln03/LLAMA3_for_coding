from classes.fixed_scheduler import FixedScheduler
from classes.concretes.sql_mixin import SqlMixin
from sqlalchemy import Column, create_engine, Table
from sqlalchemy.types import Float
from sqlalchemy.orm import registry, Session
import attr


registry = registry()

@registry.mapped
@attr.s(auto_attribs=True)
class MyClass:
    __table__ = Table(
        "my_class",
        registry.metadata,
        Column('time', Float, primary_key=True)
    )

    time: float

class MyScheduler(SqlMixin, FixedScheduler):
    def before_write(self, timestamp: float):
        return MyClass(time=timestamp)

if __name__ == "__main__":
    engine = create_engine("sqlite:///sanity.sqlite")
    registry.metadata.create_all(engine)

    with Session(engine) as session:
        with session.begin():
            scheduler = MyScheduler(1000000, MyClass.time, session)
            result = scheduler.check_and_insert()
            print(result)

        pass