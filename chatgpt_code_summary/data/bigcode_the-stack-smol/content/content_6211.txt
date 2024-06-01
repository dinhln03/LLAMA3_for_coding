from pathlib import Path
from sys import stderr

from click import command, argument
from cv2 import cv2

from life_of_photo.game_of_life import GameOfLife


WINDOW_NAME = "Life of photo"


@command()
@argument("path")
def main(path):
    path = Path(path).resolve()
    if not path.exists():
        print(f"`{path}` doesn't exist", file=stderr)
        exit(1)

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(WINDOW_NAME, image)
    simulator = GameOfLife(image)

    print("Press `q` to quit")
    print("Press `s` to save the current world into an image")

    for new_image in simulator:
        match chr(cv2.waitKeyEx()):
            case "q":
                break
            case "s":
                filename = input("Enter filename: ")
                file = path.parent / filename
                cv2.imwrite(str(file), new_image)

        cv2.imshow(WINDOW_NAME, new_image)

    cv2.destroyAllWindows()
