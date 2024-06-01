# services/resource/project/utils/enums.py


from enum import Enum


class Status(Enum):
    normal = 0
    delete = 1
    other = 2


class Scope(Enum):
    user = 'UserScope'
    admin = 'AdminScope'
