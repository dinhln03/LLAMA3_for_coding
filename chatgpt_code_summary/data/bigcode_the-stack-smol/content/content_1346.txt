# -*- coding=utf-8 -*-
import pygame
from pygame.locals import MOUSEBUTTONDOWN
from pybfcontrol.bf_common import BFControlId,BFBase, TEXT_ALIGN_LEFT,TEXT_ALIGN_MIDDLE

CLICK_EFFECT_TIME = 100
PADING = 4
class BFButton(BFBase):
    def __init__(self, parent, rect, text='Button', click=None):
        super(BFButton, self).__init__()
        self.x,self.y,self.width,self.height = rect
        self.bg_color = (225,225,225)
        self.parent = parent
        self.surface = parent.subsurface(rect)
        self.is_hover = False
        self.in_click = False
        self.click_loss_time = 0
        self.click_event_id = -1
        self.ctl_id = BFControlId().instance().get_new_id()
        self._text = text
        self._click = click
        self.init_font()

    def init_font(self):
        white = 100, 100, 100
        self.textImage = self.font.render(self._text, True, white)
        w, h = self.textImage.get_size()
        self._ty = (self.height - h) / 2
        if self._text_align == TEXT_ALIGN_LEFT:
            self._tx = PADING
        elif self._text_align == TEXT_ALIGN_MIDDLE:
            self._tx = (self.width - PADING - w) / 2
        else:
            self._tx = (self.width - PADING * 2 - w) 

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value
        self.init_font()

    @property
    def click(self):
        return self._click

    @click.setter
    def click(self, value):
        self._click = value

    def clear_hover(self):
        self.is_hover = False

    def update(self, event):
        if self.in_click and event.type == pygame.USEREVENT+1 and BFControlId().instance().click_id == self.ctl_id:
            if self._click: self._click(self)
            self.click_event_id = -1
            return True

        x, y = pygame.mouse.get_pos()
        if x > self.x and x < self.x + self.width and y > self.y and y < self.y + self.height:
            if self.panel: self.panel.clear_hover()
            self.is_hover = True
            if event.type == MOUSEBUTTONDOWN:
                pressed_array = pygame.mouse.get_pressed()
                if pressed_array[0]:
                    self.in_click = True
                    if self.panel: self.panel.clear_foucs()
                    self.click_loss_time = pygame.time.get_ticks() + CLICK_EFFECT_TIME
                    BFControlId().instance().click_id = self.ctl_id
                    pygame.time.set_timer(pygame.USEREVENT+1,CLICK_EFFECT_TIME-10)
            return True
        else:
            self.is_hover = False
        return False

    def draw(self):
        if self.in_click:
            if self.click_loss_time < pygame.time.get_ticks():
                self.in_click = False
        if not self._visible:
            return
        if self.in_click:
            r,g,b = self.bg_color
            k = 0.95
            self.surface.fill((r*k, g*k, b*k))
        else:
            self.surface.fill(self.bg_color)
        if self.is_hover:
            pygame.draw.rect(self.surface, (0,0,0), (0,0,self.width,self.height), 1)
            pygame.draw.rect(self.surface, (100,100,100), (0,0,self.width-1,self.height-1), 1)
            layers = 5
            r_step = (210-170)/layers
            g_step = (225-205)/layers
            for i in range(layers):
                pygame.draw.rect(self.surface, (170+r_step*i, 205+g_step*i, 255), (i, i, self.width - 2 - i*2, self.height - 2 - i*2), 1)
        else:
            self.surface.fill(self.bg_color)
            pygame.draw.rect(self.surface, (0,0,0), (0,0,self.width,self.height), 1)
            pygame.draw.rect(self.surface, (100,100,100), (0,0,self.width-1,self.height-1), 1)
            pygame.draw.rect(self.surface, self.bg_color, (0,0,self.width-2,self.height-2), 1)

        self.surface.blit(self.textImage, (self._tx, self._ty))

class BFButtonGroup(object):
    def __init__(self):
        self.btn_list = []

    def add_button(self, button):
        self.btn_list.append(button)

    def make_button(self, screen, rect, text='Button', click=None):
        button = BFButton(screen, rect,text=text,click=click)
        self.add_button(button)

    def update(self, event):
        for button in self.btn_list: button.update(event)

    def draw(self):
        for button in self.btn_list: button.draw()
