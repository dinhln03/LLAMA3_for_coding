import app.constants as const


class Config:
    def __init__(self, shape=const.DEFAULT_CONFIG_SHAPE,
                 size=const.DEFAULT_CONFIG_SIZE,
                 max_thick=const.MAXIMUM_CONFIG_THICKNESS,
                 min_thick=const.MINIMUM_CONFIG_THICKNESS,
                 use_border=const.DEFAULT_CONFIG_BORDER,
                 border_thick=const.DEFAULT_CONFIG_BORDER_THICKNESS,
                 curve=const.DEFAULT_CONFIG_CURVE,
                 stl_format=const.DEFAULT_CONFIG_FORMAT):
        self.shape = shape
        self.size = size
        self.max_thickness = max_thick
        self.min_thickness = min_thick
        self.use_border = use_border
        self.border_thickness = border_thick
        self.curve = curve
        self.format = stl_format

    def get_config(self):
        return self
