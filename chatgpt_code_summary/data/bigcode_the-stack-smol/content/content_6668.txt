from typing import List

from pdip.data import Entity
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from process.domain.base import Base
from process.domain.base.operation.DataOperationBase import DataOperationBase
from process.domain.operation.DataOperationContact import DataOperationContact
from process.domain.operation.DataOperationIntegration import DataOperationIntegration
from process.domain.operation.DataOperationJob import DataOperationJob


class DataOperation(DataOperationBase, Entity, Base):
    __tablename__ = "DataOperation"
    __table_args__ = {"schema": "Operation"}
    DefinitionId = Column(Integer, ForeignKey('Operation.Definition.Id'))
    Name = Column(String(100), index=True, unique=False, nullable=False)
    Definition = relationship("Definition", back_populates="DataOperations")
    DataOperationJobs: List[DataOperationJob] = relationship("DataOperationJob", back_populates="DataOperation")
    Integrations: List[DataOperationIntegration] = relationship("DataOperationIntegration",
                                                                back_populates="DataOperation")
    Contacts: List[DataOperationContact] = relationship("DataOperationContact",
                                                        back_populates="DataOperation")
