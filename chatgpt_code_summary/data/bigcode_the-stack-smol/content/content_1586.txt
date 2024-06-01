__author__ = 'Burgos, Agustin - Schelotto, Jorge'
# -*- coding: utf-8 -*-

# Copyright 2018 autors: Burgos Agustin, Schelotto Jorge
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#  TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import pygame

class Palabras(pygame.sprite.Sprite):
    def __init__(self, ruta, nombre, x, y):
        super().__init__()
        self.__palabra = nombre
        self.__click = False
        self.image = pygame.image.load(ruta).convert_alpha()
        self.rect = self.image.get_rect()
        self.collide = False
        self.posX = x
        self.posY = y

    def getPosX(self):
        return self.posX

    def getPosY(self):
        return self.posY

    def getPalabra(self):
        return self.__palabra

    def getPalabraImagen(self):
        return self.image

    def setClick(self, bool):
        self.__click = bool

    def getClick(self):
        return self.__click

    def getRect(self):
        return self.rect

    def colli(self, x, y):

        if x > 20:
            # Achica la imagen
            center = self.rect.center
            x = x - 1
            y = y - 1
            self.image = pygame.transform.scale(self.image, (x, y))
            self.rect = self.image.get_rect()
            self.rect.center = center
            self.image = pygame.transform.rotozoom(self.image, -90, 0.8)
        elif x <= 20:
            # Para que no de x < 0
            center = self.rect.center
            self.image = pygame.transform.scale(self.image, (0, 0))
            self.rect = self.image.get_rect()
            self.rect.center = center
            self.image = pygame.transform.rotozoom(self.image, -90, 0.5)


    def update(self,surface):
        """Controla los eventos y coliciones de los sprites Palabras"""
        if not self.getClick() and not self.collide:
            self.rect.center = (self.posX, self.posY)

        if self.getClick():
            #Si se hace click en la imagen
            self.rect.center = pygame.mouse.get_pos()

        if self.collide:
            # Si hay colision
            x = self.image.get_rect().size[0]
            y = self.image.get_rect().size[1]
            self.colli(x,y)
            # Saca la imagen de la zona de colición.
            if self.image.get_rect().size[0] <= 20:
                self.rect.center = (0,0)

        surface.blit(self.getPalabraImagen(), self.getRect())


