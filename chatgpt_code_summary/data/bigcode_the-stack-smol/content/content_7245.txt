class RGB:
    def __init__(self, red=0, green=0, blue=0):
        self.r = 0
        self.g = 0
        self.b = 0
        self.red = red
        self.green = green
        self.blue = blue

    @property
    def red(self):
        return self.r

    @red.setter
    def red(self, value):
        if isinstance(value, int):
            if 0 <= value <= 255:
                self.r = value
            else:
                raise ValueError("Int value of R is out of range 0-255: {0}".format(value))
        elif isinstance(value, float):
            if 0.0 <= value <= 1.0:
                self.r = int(value*255)
            else:
                raise ValueError("Float value of R is out of range 0.0-1.0: {0}".format(value))
        else:
            raise TypeError("Color must be int or float")

    @property
    def red_float(self):
        return self.r/255.0

    @property
    def green(self):
        return self.g

    @green.setter
    def green(self, value):
        if isinstance(value, int):
            if 0 <= value <= 255:
                self.g = value
            else:
                raise ValueError("Int value of G is out of range 0-255: {0}".format(value))
        elif isinstance(value, float):
            if 0.0 <= value <= 1.0:
                self.g = int(value*255)
            else:
                raise ValueError("Float value of G is out of range 0.0-1.0: {0}".format(value))
        else:
            raise TypeError("Color must be int or float")

    @property
    def green_float(self):
        return self.g/255.0

    @property
    def blue(self):
        return self.b

    @blue.setter
    def blue(self, value):
        if isinstance(value, int):
            if 0 <= value <= 255:
                self.b = value
            else:
                raise ValueError("Int value of B is out of range 0-255: {0}".format(value))
        elif isinstance(value, float):
            if 0.0 <= value <= 1.0:
                self.b = int(value*255)
            else:
                raise ValueError("Float value of B is out of range 0.0-1.0: {0}".format(value))
        else:
            raise TypeError("Color must be int or float")

    @property
    def blue_float(self):
        return self.b/255.0

