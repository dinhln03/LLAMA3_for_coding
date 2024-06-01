import base64


def decode_img(img_string):
    img_data = base64.b64decode(img_string)
    filename = "temp_img.jpg"
    with open(filename, "wb") as f:
        f.write(img_data)
