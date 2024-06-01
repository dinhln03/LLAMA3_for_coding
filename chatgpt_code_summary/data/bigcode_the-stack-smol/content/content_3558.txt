from pydantic import BaseModel, HttpUrl
from typing import List

class RegisterHostModel(BaseModel):
    name: str
    url:  HttpUrl
    ports: List[int]
