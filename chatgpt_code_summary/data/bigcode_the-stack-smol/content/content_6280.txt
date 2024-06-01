from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, LargeBinary, Float, UniqueConstraint
from sqlalchemy.orm import relationship, backref
from datetime import datetime

from conductor.app.db.base_class import Base


class DiscoveryResult(Base):
    __tablename__ = "discovery_results"
    __table_args__ = (
        # this can be db.PrimaryKeyConstraint if you want it to be a primary key
        UniqueConstraint('train_id', 'station_id'),
    )
    id = Column(Integer, primary_key=True, index=True)
    train_id = Column(Integer, ForeignKey("trains.id"))
    station_id = Column(Integer, ForeignKey("stations.id"))
    results = Column(String)
    created_at = Column(DateTime, default=datetime.now())
