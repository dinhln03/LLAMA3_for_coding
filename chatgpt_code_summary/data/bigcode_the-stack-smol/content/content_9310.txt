"""Tests for the smarttub integration."""

from datetime import timedelta

from openpeerpower.components.smarttub.const import SCAN_INTERVAL
from openpeerpower.util import dt

from tests.common import async_fire_time_changed


async def trigger_update(opp):
    """Trigger a polling update by moving time forward."""
    new_time = dt.utcnow() + timedelta(seconds=SCAN_INTERVAL + 1)
    async_fire_time_changed(opp, new_time)
    await opp.async_block_till_done()
