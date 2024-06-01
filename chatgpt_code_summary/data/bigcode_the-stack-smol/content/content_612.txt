from . import GB, HDB
from typing import Literal


class Client:
    def __init__(self, t: Literal["gb", "hbba", "dbba"]):
        self.type = t

    def create(self):
        if self.type == "gb":
            return GB()
        elif self.type == "hb":
            return HDB("hbba")
        elif self.type == "db":
            return HDB("dbba")
