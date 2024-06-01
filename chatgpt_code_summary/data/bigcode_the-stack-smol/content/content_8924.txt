import pygame
import sys 
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import scene.scene
import scene.shape

pygame.init() 
pygame.display.set_caption('draw alpha')

screen = pygame.display.set_mode((1280, 768))
running = True 

s1 = scene.scene.Scene(1280, 768, 0, 0, 1280, 768)

c1 = scene.shape.Circle(100, 100, (255, 0, 0, 128), 30, 1)
c1.transform.move(200, 100)
c2 = scene.shape.Circle(600, 600, (255, 0, 0, 128), 300, 1)
c2.transform.move(200, 100)

r1 = scene.shape.Rect(100, 300, (0, 255, 0, 128), 1)
r1.transform.move(300, 100)

s1.attach('circle_01', c1)
s1.attach('circle_02', c2)
s1.attach('rect_01', r1)

s1.zoom(1)
s1.pan(0, -500)
#s1.locate(300, 300)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(scene.shape.Color.WHITE)
    s1.draw(screen)
    cir = s1.find('circle_01')
    cir.transform.move(0.1, 0.1)
    r = s1.find('rect_01')
    r.transform.move(0.0, 0.1)
    s1.pan(-0.05, 0.1)
    s1.zoom(0.999)
    pygame.display.flip()

