from typing import Optional, List

from discord import Member

from config import PERMITTED_ROLE_IDS
from src.util.includes import includes


def is_authorized_member(member: Member, authorized_ids: Optional[List[int]] = None) -> bool:
    if not authorized_ids:
        authorized_ids = PERMITTED_ROLE_IDS

    member_role_ids: List[int] = [
        role.id for role in member.roles
    ]

    if includes(member_role_ids, authorized_ids):
        return True

    return False
