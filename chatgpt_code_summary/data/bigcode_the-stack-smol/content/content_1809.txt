
from equinox.models import Model,cleanup


import glm

from random import random
from .glutils import bindIndicesToBuffer, storeDataInVBO,createVAO,unbindVAO


class Terrain(Model):
    
    def __init__(self, n_vertex):
        self.vertices = (
            -1.0, 0.0, 1.0,
            -1.0, 0.0, -1.0,
             1.0, 0.0, -1.0,
             1.0, 0.0,  1.0,
        )
        self.normals = (
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0
        )
        self.indices = (
            0,1,2,
            2,3,0
        )
        
    


