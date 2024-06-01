#!/usr/bin/env python
# -*- coding: utf-8 -*-
import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, Request

from epicteller.core.controller import campaign as campaign_ctl
from epicteller.core.controller import room as room_ctl
from epicteller.core.error.base import NotFoundError
from epicteller.web.controller.paging import generate_paging_info
from epicteller.web.fetcher import room as room_fetcher
from epicteller.web.model import PagingResponse
from epicteller.web.model.campaign import Campaign
from epicteller.web.model.room import Room


router = APIRouter()


async def prepare(url_token: str):
    room = await room_ctl.get_room(url_token=url_token)
    if not room or room.is_removed:
        raise NotFoundError()
    return room


@router.get('/rooms/{url_token}', response_model=Room, response_model_exclude_none=True)
async def get_room(room: Room = Depends(prepare)):
    web_room = await room_fetcher.fetch_room(room)
    return web_room


@router.get('/rooms/{url_token}/campaigns', response_model=PagingResponse[Campaign], response_model_exclude_none=True)
async def get_room_campaigns(r: Request, room: Room = Depends(prepare), after: Optional[str] = None,
                             offset: Optional[int] = 0, limit: Optional[int] = 20):
    after_id = 0
    if after_campaign := await campaign_ctl.get_campaign(url_token=after):
        after_id = after_campaign.id
    total, campaigns = await asyncio.gather(
        campaign_ctl.get_campaign_count_by_room(room),
        campaign_ctl.get_campaigns_by_room(room, after_id, limit),
    )
    paging_info = await generate_paging_info(r,
                                             total=total,
                                             after=campaigns[-1].id if len(campaigns) else None,
                                             offset=offset,
                                             limit=limit)
    return PagingResponse[Campaign](data=campaigns, paging=paging_info)
