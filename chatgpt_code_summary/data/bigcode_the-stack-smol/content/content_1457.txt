
class Beam(object):

    def __init__(self):
        super().__init__()

    def get_number_of_rays(self):
        raise NotImplementedError("method is abstract")

    def get_rays(self):
        raise NotImplementedError("method is abstract")

    def get_ray(self, ray_index):
        raise NotImplementedError("method is abstract")

    def duplicate(self):
        raise NotImplementedError("method is abstract")

    def merge(self, other_beam):
        raise NotImplementedError("method is abstract")

