from PIL import Image


def img_to_binary(img: Image, min_value=100) -> Image:
    return img.convert('L').point(lambda p: p > min_value and 255).convert('1')
