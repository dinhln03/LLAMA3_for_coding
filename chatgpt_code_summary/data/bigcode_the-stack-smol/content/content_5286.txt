from json import load
from typing import Union

import pygame as pg
from pygame import Surface, event
from pygame.display import set_mode, set_caption, set_icon, get_surface, update
from pygame.key import get_pressed as get_key_pressed
from pygame.mouse import get_pressed as get_mouse_pressed
from pygame.time import Clock, get_ticks
from pygame.transform import scale

from source.imgs.sprite_sheet import load_sprite_sheet
from source.scripts.scenes import Game, Menu, Shop, Scene
from source.sounds.manager import get_song


class App:
    all_keycodes = tuple(getattr(pg.constants, key_str) for key_str in
                         filter(lambda k: k.startswith("K_"), dir(pg.constants)))

    def __init__(self, config: dict[str, Union[int, str]] = None):
        # ## load
        # # load config
        if config is None:
            with open("source/config.json") as file:
                config = load(file)

        # get config
        self.width: int = ...
        self.height: int = ...
        self.fps: int = ...
        self.start_scene: str = ...
        self.volume: int = ...

        for name in ["width", "height", "fps", "start_scene", "volume"]:
            setattr(self, name, config[name])

        self.done = True
        self.clock = Clock()

        # # load images
        self.bg_sprite_sheet = load_sprite_sheet("bg_imgs")
        # ## create and initialize
        # # create music

        self.bgm_intro = get_song("intro.wav")
        self.bgm = get_song("bgm.wav")
        self.bgm_is_running = False

        self.bgm_intro.set_volume(self.volume)
        self.bgm.set_volume(self.volume)

        # # create scenes
        self.name = "OREO Clicker"
        self.screen: Surface

        self.game = Game(self)
        self.menu = Menu(self)
        self.shop = Shop(self)

        self._scene: Scene = getattr(self, self.start_scene, "menu")

        # initialize scenes
        self.game.initialize()
        self.menu.initialize()
        self.shop.initialize()

    @property
    def scene(self) -> Scene:
        return self._scene

    @scene.setter
    def scene(self, value: Scene):
        self._scene = value
        self.update_screen()

    @property
    def scene_scale(self):
        return self.scene.settings["scale"][0] / self.scene.settings["size"][0], \
               self.scene.settings["scale"][1] / self.scene.settings["size"][1]

    def update_screen(self):
        set_mode(self.scene.settings["scale"])
        set_caption(self.scene.settings["title"])
        if self.scene.settings["icon"]:
            set_icon(self.scene.settings["icon"])

        # noinspection PyAttributeOutsideInit
        self.screen = get_surface()

    @property
    def scene_screen(self):
        return Surface(self.game.settings["size"])

    def draw(self):
        self.screen.blit(scale(self.scene.draw(), self.scene.settings["scale"]), (0, 0))
        update()

    def update(self):
        self.scene.update()

    def run(self):
        self.bgm_intro.play()
        self.done = False
        self.update_screen()
        while not self.done:
            if get_ticks() // (self.fps * 20) >= int(self.bgm_intro.get_length()) and not self.bgm_is_running:
                self.bgm_intro.stop()
                self.bgm.play(-1)
                self.bgm_is_running = True
            self.draw()
            self.handle_events()
            self.handle_input()
            self.clock.tick(self.fps)
            self.update()

    def handle_events(self):
        for event_ in event.get():
            if event_.type == pg.QUIT:
                self.done = True
                break
            if "events_filter" not in self.scene.settings or event_.type in self.scene.settings["events_filter"]:
                self.scene.handle_event(event)

    def handle_input(self):
        self.handle_mouse_press()
        keys_pressed = get_key_pressed()
        for keycode in self.all_keycodes:
            if keys_pressed[keycode]:
                self.scene.handle_input(keycode)

    def handle_mouse_press(self):
        pressed = get_mouse_pressed(3)
        mouse_pos = self.get_mouse_pos()
        for key in range(3):
            if pressed[key]:
                self.scene.handle_mouse_press(key, mouse_pos)

    def get_mouse_pos(self):
        return pg.mouse.get_pos()[0] // self.scene_scale[0], \
               pg.mouse.get_pos()[1] // self.scene_scale[1]
