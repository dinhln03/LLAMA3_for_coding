from typing import Dict

from handler import Context, Arguments, CommandResult
from rpg.items import Item
from utils.formatting import codeblock
from utils.command_helpers import get_author_player


async def run(ctx: Context, args: Arguments) -> CommandResult:
    player = await get_author_player(ctx)

    if player.inventory.size:
        counts: Dict[Item, int] = {}
        for item in player.inventory:
            counts[item] = counts.get(item, 0) + 1

        inventory = "\n".join(
            f"{item}{' x ' + str(count) if count > 1 else ''}"
            for item, count in counts.items()
        )
    else:
        inventory = "Ваш инвентарь пуст"

    equipment_item_map = [
        (slot, getattr(player.equipment, slot)) for slot in player.equipment._slots
    ]

    equipment = "\n".join(f"{slot:>10}: {item}" for (slot, item) in equipment_item_map)

    return codeblock(f"Экипировка:\n\n{equipment}\n\nИнвентарь:\n\n{inventory}")
