#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Display a calendar populated from google calendar data on an inky display."""

from PIL import Image, ImageDraw  # type: ignore

# from typing import Tuple
# import time


def draw_what_sheet(image: Image.Image) -> None:
    """Draw a calendar page for a WHAT display.

    Args:
        image: The image to be drawn on to

    """
    draw = ImageDraw.Draw(image)
    # draw.rectangle([(7, 3), (392, 296)], outline=1)
    draw.line([(7, 3), (392, 3)], fill=1)
    for line in range(8):
        draw.line([(line * 55 + 7, 3), (line * 55 + 7, 296)], fill=1)
    for line in range(7):
        draw.line([(7, line * 45 + 26), (392, line * 45 + 26)], fill=1)


if __name__ == "__main__":
    palette = 3 * [255]
    palette += 3 * [0]
    palette += [255, 0, 0]
    palette += 759 * [0]

    img = Image.new("P", (400, 300), color=0)

    draw_what_sheet(img)

    img.putpalette(palette)
    img.save("calendar.png")
    try:
        from inky import InkyWHAT  # type: ignore
    except RuntimeError:
        pass
    except ModuleNotFoundError:
        pass
    else:
        inky_display = InkyWHAT("red")
        inky_display.set_image(img)
        inky_display.set_border(inky_display.BLACK)
        inky_display.show()
