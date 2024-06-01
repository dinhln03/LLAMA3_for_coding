import keyboard
import settings
from key_sender import *
import utils


class HotKey(object):
    def __init__(self):
        pass

    def regist_hotkey(self, hotkey_group, queue_h):
        if settings.test:
            keyboard.add_hotkey('F10', self.f10_fun)
            keyboard.add_hotkey('F11', self.f11_fun)
            keyboard.add_hotkey('F12', self.f12_fun)
        elif hotkey_group == settings.KeyGroupEnum.modau:
            keyboard.add_hotkey('F10', self.modau_f10_fun)
            keyboard.add_hotkey('F11', self.modau_f11_fun)
            keyboard.add_hotkey('F12', self.modau_f12_fun)

        keyboard.wait('esc')
        queue_h.put('end')

    @staticmethod
    def send_key(key):
        # time.sleep(0.1)
        key_press(key)

    def f10_fun(self):
        # self.get_foreground_title()
        self.send_key(Key['down_arrow'])
        self.send_key(Key['up_arrow'])
        self.send_key(Key['spacebar'])

    def f11_fun(self):
        self.send_key(Key['up_arrow'])
        self.send_key(Key['up_arrow'])
        self.send_key(Key['spacebar'])

    def f12_fun(self):
        print(utils.get_foreground_title())
        # self.send_key(Key['up_arrow'])
        # self.send_key(Key['right_arrow'])
        # self.send_key(Key['spacebar'])

    # 魔道
    def modau_f10_fun(self):
        self.send_key(Key['down_arrow'])
        self.send_key(Key['up_arrow'])
        self.send_key(Key['spacebar'])

    def modau_f11_fun(self):
        self.send_key(Key['up_arrow'])
        self.send_key(Key['up_arrow'])
        self.send_key(Key['spacebar'])

    def modau_f12_fun(self):
        self.send_key(Key['up_arrow'])
        self.send_key(Key['right_arrow'])
        self.send_key(Key['spacebar'])
