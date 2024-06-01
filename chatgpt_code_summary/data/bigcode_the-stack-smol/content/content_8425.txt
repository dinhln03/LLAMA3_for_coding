from datetime import datetime
from infrastructure.cqrs.decorators.dtoclass import dtoclass


@dtoclass
class GetDataOperationJobListDto:
    Id: int = None
    JobId: int = None
    DataOperationId: str = None
    DataOperationName: str = None
    Cron: str = None
    StartDate: datetime = None
    EndDate: datetime = None
    NextRunTime: datetime = None
    CreationDate: datetime = None
    LastUpdatedDate: datetime = None
    IsDeleted: int = None
