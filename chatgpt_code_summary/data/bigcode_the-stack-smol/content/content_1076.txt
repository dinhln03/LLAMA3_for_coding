from __future__ import annotations

import numpy as np
import tcod

import g
import game.constants
import game.engine
import game.game_map
import game.render_functions
from game.tiles import tile_graphics


def render_map(console: tcod.Console, gamemap: game.game_map.GameMap) -> None:
    # The default graphics are of tiles that are visible.
    light = tile_graphics[gamemap.tiles]
    light[gamemap.fire > 0] = (ord("^"), (255, 255, 255), (0xCC, 0x22, 0))

    # Apply effects to create a darkened map of tile graphics.
    dark = gamemap.memory.copy()
    dark["fg"] //= 2
    dark["bg"] //= 8

    visible = gamemap.visible
    if g.fullbright:
        visible = np.ones_like(visible)

    for entity in sorted(gamemap.entities, key=lambda x: x.render_order.value):
        if not visible[entity.x, entity.y]:
            continue  # Skip entities that are not in the FOV.
        light[entity.x, entity.y]["ch"] = ord(entity.char)
        light[entity.x, entity.y]["fg"] = entity.color

    console.rgb[0 : gamemap.width, 0 : gamemap.height] = np.select(
        condlist=[visible, gamemap.explored],
        choicelist=[light, dark],
        default=dark,
    )

    for entity in sorted(gamemap.entities, key=lambda x: x.render_order.value):
        if not visible[entity.x, entity.y]:
            continue  # Skip entities that are not in the FOV.
        console.print(entity.x, entity.y, entity.char, fg=entity.color)

    visible.choose((gamemap.memory, light), out=gamemap.memory)


def render_ui(console: tcod.Console, engine: game.engine.Engine) -> None:
    UI_WIDTH = game.constants.ui_width
    UI_LEFT = console.width - UI_WIDTH
    LOG_HEIGHT = console.height - 8

    engine.message_log.render(
        console=console, x=UI_LEFT, y=console.height - LOG_HEIGHT, width=UI_WIDTH, height=LOG_HEIGHT
    )
    console.draw_rect(UI_LEFT, 0, UI_WIDTH, 2, 0x20, (0xFF, 0xFF, 0xFF), (0, 0, 0))

    game.render_functions.render_bar(
        console=console,
        x=UI_LEFT,
        y=0,
        current_value=engine.player.fighter.hp,
        maximum_value=engine.player.fighter.max_hp,
        total_width=UI_WIDTH,
    )

    game.render_functions.render_names_at_mouse_location(console=console, x=UI_LEFT, y=1, engine=engine)

    if g.mouse_pos:
        console.rgb[g.mouse_pos]["fg"] = (0, 0, 0)
        console.rgb[g.mouse_pos]["bg"] = (255, 255, 255)
        if g.fullbright or engine.game_map.visible[g.mouse_pos]:
            console.print(
                UI_LEFT,
                2,
                f"Fire={engine.game_map.fire[g.mouse_pos]}, Heat={engine.game_map.heat[g.mouse_pos]}, "
                f"Smoke={engine.game_map.smoke[g.mouse_pos]},\nFuel={engine.game_map.fuel[g.mouse_pos]}",
            )
