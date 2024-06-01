import os

import moderngl
import numpy as np
from objloader import Obj
from PIL import Image
from pyrr import Matrix44

import data
from window import Example, run_example


class CrateExample(Example):
    title = "Crate"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_vert;
                in vec3 in_norm;
                in vec2 in_text;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Light;
                uniform sampler2D Texture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    f_color = vec4(texture(Texture, v_text).rgb * lum, 1.0);
                }
            ''',
        )

        self.mvp = self.prog['Mvp']
        self.light = self.prog['Light']

        obj = Obj.open(data.find('crate.obj'))
        img = Image.open(data.find('crate.png')).transpose(Image.FLIP_TOP_BOTTOM).convert('RGB')
        self.texture = self.ctx.texture(img.size, 3, img.tobytes())
        self.texture.use()

        self.vbo = self.ctx.buffer(obj.pack('vx vy vz nx ny nz tx ty'))
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_norm', 'in_text')

    def render(self, time, frame_time):
        angle = time
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        camera_pos = (np.cos(angle) * 5.0, np.sin(angle) * 5.0, 2.0)

        proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
        lookat = Matrix44.look_at(
            camera_pos,
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
        )

        self.mvp.write((proj * lookat).astype('f4').tobytes())
        self.light.value = camera_pos
        self.vao.render()


if __name__ == '__main__':
    run_example(CrateExample)
