from operator import ge
from typing import List, Optional, Tuple  # Dict,
from fastapi import FastAPI, HTTPException, Depends, Query, status
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi import Request  # , Response

# from fastapi.responses import JSONResponse
# from pymongo.common import validate_server_api_or_none

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase


import datetime as dt

from app.core.config import get_logger


from todoer_api.model import (
    Task,
    TodoerInfo,
    TaskCreate,
    TaskUpdate,
    TaskPartialUpdate,
    ObjectId,
)
from todoer_api.data_layer import (
    TaskDatabase,
    DataLayerException,
    database_factory,
)
from todoer_api import __version__, __service_name__

logger = get_logger("todoer")
BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


# ------------------------------------------------------------------------------
app = FastAPI()

# ------------------------------------------------------------------------------
# task_db: TaskDatabase = database_factory("mongo")  # None
task_db: TaskDatabase = database_factory("mongo")


# async def build_database() -> TaskDatabase:
#     return database_factory("mongo")


async def get_database() -> TaskDatabase:
    # !!! for some reason when trying to saccess the DB via the data layer
    # it creates an error: attached to a different loop
    # don't know why left it to a local variable in main
    # global task_db
    return task_db


def pagination(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=0),
) -> Tuple[int, int]:
    capped_limit = min(100, limit)
    return (skip, capped_limit)


async def get_task_or_404(task_key: str, database=Depends(get_database)) -> Task:
    try:
        return await database.get(task_key)
    except DataLayerException:
        raise HTTPException(status_code=404, detail=f"Task {task_key} not found")


# ------------------------------------------------------------------------------

# database_builder=Depends(build_database)
@app.on_event("startup")
async def startup():
    global task_db
    # await
    task_db = database_factory("mongo")  #


@app.on_event("shutdown")
async def shutdown():
    global task_db
    del task_db
    task_db = None


@app.get("/todoer/v1/tasks", status_code=200)
async def root(
    request: Request,
    database=Depends(get_database),
    pagination: Tuple[int, int] = Depends(pagination),
) -> dict:  # 2
    """
    GET tasks as html page
    """
    tasks = await database.get_all(*pagination)
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "tasks": tasks},
    )


@app.get("/todoer/api/v1/ping")
async def model_ping():
    return {"ping": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/todoer/api/v1/info", response_model=TodoerInfo)
async def model_info(database=Depends(get_database)) -> TodoerInfo:
    logger.info(f"get info")
    return TodoerInfo(
        timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        service=__service_name__,
        data_source=database.db_type,
        version=__version__,
    )


@app.get("/todoer/api/v1/tests/{test_id}")
async def test(test_id: int, qry: Optional[str] = None):
    logger.info(f"in test id={test_id} qry={qry}")
    return {"test_id": test_id, "q": qry}


@app.get("/todoer/api/v1/tasks")
async def get_tasks(
    pagination: Tuple[int, int] = Depends(pagination), database=Depends(get_database)
) -> List[Task]:
    return await database.get_all(*pagination)


@app.get("/todoer/api/v1/tasks/{task_key}", response_model=Task)
async def get_task_id(task: Task = Depends(get_task_or_404)) -> Task:
    return task


@app.post("/todoer/api/v1/tasks", status_code=201, response_model=Task)
async def create_task(task: TaskCreate, database=Depends(get_database)) -> Task:
    try:
        logger.info(f"request to create task in project {task.project}")
        added_task = await database.add(task)
        return added_task
    except DataLayerException:
        raise HTTPException(
            status_code=409, detail=f"Adding task key {task.key} failed, already exists"
        )


@app.put("/todoer/api/v1/tasks/{task_key}", response_model=Task)
async def update_task(
    task_key: str, task: TaskUpdate, database=Depends(get_database)
) -> Task:
    try:
        logger.info(f"request to update task: {task_key}")
        udp_task = await database.update(task_key, task)
        return udp_task
    except DataLayerException:
        raise HTTPException(status_code=404, detail=f"Task {task_key} not found")


@app.patch("/todoer/api/v1/tasks/{task_key}", response_model=Task)
async def patch_task(
    task_key: str, task: TaskPartialUpdate, database=Depends(get_database)
) -> Task:
    try:
        logger.info(f"request to patch task: {task_key}")
        return await database.update(task_key, task)
    except DataLayerException:
        raise HTTPException(status_code=404, detail=f"Task {task_key} not found")


@app.delete("/todoer/api/v1/tasks/{task_key}", status_code=204)
async def del_task(task_key: str, database=Depends(get_database)) -> None:
    try:
        logger.info(f"request to delete task: {task_key}")
        await database.delete(task_key)
    except DataLayerException:
        raise HTTPException(status_code=404, detail=f"Delete task {task_key} not found")


@app.delete("/todoer/admin/v1/tasks", status_code=204)
async def del_all_task(database=Depends(get_database)):
    try:
        logger.info("request to delete all tasks")
        await database.delete_all()
    except DataLayerException:
        raise HTTPException(status_code=404, detail=f"Failed to delete all tasks")
