from __future__ import annotations
import argparse
import logging
from typing import TextIO


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType('r'),
                        metavar="PUZZLE_INPUT")
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args(args)
    return args


def init_logging(debug=False):
    msg_format = '%(asctime)s %(levelname)s %(message)s'
    date_format = '%m/%d/%Y %H:%M:%S'
    level = logging.INFO
    if debug:
        level = logging.DEBUG
    logging.basicConfig(format=msg_format, datefmt=date_format, level=level)


class Image:

    def __init__(self, pixels: dict[tuple[int, int], str], void_pixel: str):
        self.pixels = pixels
        self.void_pixel = void_pixel

    def __getitem__(self, key: tuple[int, int]) -> str:
        try:
            return self.pixels[key]
        except KeyError:
            return self.void_pixel

    @staticmethod
    def from_grid(grid: list[list[str]]) -> Image:
        pixels = Image.grid2pixel(grid)
        return Image(pixels, '.')

    @staticmethod
    def grid2pixel(grid: list[list[str]]) -> dict[tuple[int, int], str]:
        image = {}
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                image[(x, y)] = grid[y][x]
        return image

    @staticmethod
    def neighbors(pixel: tuple[int, int]) -> list[tuple[int, int]]:
        x = pixel[0]
        y = pixel[1]
        return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                (x - 1, y),     (x, y),     (x + 1, y),
                (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]

    def pixel2idx(self, pixel: str) -> int:
        bin_rep = pixel.replace('#', '1').replace('.', '0')
        return int(bin_rep, base=2)

    def enhance_pixel(self, iea: str, pixel: tuple[int, int]) -> str:
        surround = [self[n] for n in self.neighbors(pixel)]
        idx = self.pixel2idx(''.join(surround))
        return iea[idx]

    def bounds(self) -> tuple[int, ...]:
        x_values = [p[0] for p in self.pixels]
        y_values = [p[1] for p in self.pixels]
        return min(x_values), min(y_values), max(x_values), max(y_values)

    def enhance(self, iea: str) -> Image:
        new_pixels = {}
        min_x, min_y, max_x, max_y = self.bounds()
        for x in range(min_x - 2, max_x + 2):
            for y in range(min_y - 2, max_y + 2):
                new_pixels[(x, y)] = self.enhance_pixel(iea, (x, y))
        void_pixel = iea[self.pixel2idx(self.void_pixel * 9)]
        return Image(new_pixels, void_pixel)

    def lit_count(self):
        return len([v for v in self.pixels.values() if v == '#'])


def load_input(fp: TextIO):
    data = fp.read().strip().split('\n\n')
    iea = data[0]
    assert len(iea) == 512
    grid = []
    for line in data[1].strip().split('\n'):
        grid.append(list(line))
    image = Image.from_grid(grid)
    return iea, image


def puzzle1(iea: str, image: Image) -> int:
    for i in range(2):
        image = image.enhance(iea)
    return image.lit_count()


def puzzle2(iea, image) -> int:
    for i in range(50):
        image = image.enhance(iea)
    return image.lit_count()


def main(argv=None):
    args = parse_args(argv)

    # Init logging
    init_logging(args.debug)

    iea, image = load_input(args.input)
    answer = puzzle1(iea, image)
    logging.info('Puzzle 1: %d', answer)
    answer = puzzle2(iea, image)
    logging.info('Puzzle 2: %d', answer)


if __name__ == '__main__':
    main()
