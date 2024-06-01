import time
from dataclasses import dataclass

import jwt

from util.enum_util import EnumBase


class Role(EnumBase):
    USER = "USER"
    ADMIN = "ADMIN"


class Perm(EnumBase):
    NONE = "NONE"
    READ_MSG = "READ_MSG"
    WRITE_MSG = "WRITE_MSG"


@dataclass
class Identity:
    user: str
    role: Role.to_enum()

    perm_mapping = {
        Role.USER: [Perm.READ_MSG],
        Role.ADMIN: [Perm.READ_MSG, Perm.WRITE_MSG],
    }

    def has_permission(self, perm: Perm) -> bool:
        return perm in self.perm_mapping[self.role]


class JWTAuthenticator(object):
    ACCESS_JWT_ALGORITHM = "HS256"

    @classmethod
    def dump_access_token(cls, key: str, identity: Identity, exp: int) -> str:
        current_ts = int(time.time())
        return jwt.encode(
            payload=dict(
                user=identity.user,
                role=identity.role,
                nbf=current_ts - 300,  # not before
                exp=current_ts + exp,
            ),
            key=key,
            algorithm=cls.ACCESS_JWT_ALGORITHM,
        )

    @classmethod
    def load_access_token(cls, key: str, access_token: str) -> Identity:
        payload = jwt.decode(
            jwt=access_token, key=key, algorithms=[cls.ACCESS_JWT_ALGORITHM]
        )
        return Identity(user=payload["user"], role=payload["role"])
