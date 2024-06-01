
"""

Création des images pour la tâche de détermination numérique
(pour évaluer l'impact de la configuration sur le subitizing)
Victor ANTOINE - victor.antoine@ens.fr 

"""

import pygame
from random import sample
from numpy import random, sort
from os import path
from itertools import product

W, H = 960, 540

pygame.init()
screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
screen.fill((0, 0, 0))

#création des images en disposition aléatoire
origin_x, origin_y = random.randint(50, 910), random.randint(50, 490)
list_coord_random_x = list_coords_random_y = []

def create_liste_coord_random(axe, origin):
    coord1 = coord2 = origin
    liste = []
    liste.append(origin)
    while coord1 <= axe - 160:
        coord1 += 80
        liste.append(coord1)    
    while coord2 >= 110:
        coord2 -= 80
        liste.append(coord2)    
    liste = list(sort(liste))
    return liste

list_coord_random_x = create_liste_coord_random(W, origin_x)
list_coord_random_y = create_liste_coord_random(H, origin_y)
system_coord_random = list(product(list_coord_random_x, list_coord_random_y))

for version in list(range(1, 11)):
    for points_number in list(range(1, 11)):
        screen.fill((0, 0, 0))
        for (x, y) in sample(system_coord_random, points_number):
            pygame.draw.circle(screen, (255, 255, 255), (x, y), 30, 0)    
        pygame.image.save(screen, path.join("pictures", "random", \
                          str(points_number) + "_" + str(version) + ".png"))    

#création des images en dispostion configurationnelle
def create_slot_coord_config(top, left):
    liste_coord = []
    for position in [(1, 1), (3, 1), (2, 2), (1, 3), (3, 3)]:
        liste_coord.append((top + position[0] * ((W - 270)/8),\
                           left + position[1] * ((H - 270)/4)))
    return liste_coord

coord_left_side = create_slot_coord_config(130, 130)
coord_mid_side = create_slot_coord_config(303, 130)
coord_right_side = create_slot_coord_config(475, 130)

system_coord_config = []
position = [[2], [1, 3], [1, 2, 3], [0, 1, 3, 4], [0, 1, 2, 3, 4]]

for number in range(1, 11):
    list_coord = []
    
    if number <= 5:
        positions = position[number-1]
        for circle in positions:
            list_coord.append(coord_mid_side[circle])
        system_coord_config.append(list_coord)
        
    else:
        for circle in position[4]:
            list_coord.append(coord_left_side[circle])
        positions = position[number-6]
        for circle in positions:
            list_coord.append(coord_right_side[circle])
        system_coord_config.append(list_coord)

number_index = 1
for number in system_coord_config:
    screen.fill((0, 0, 0))
    for (x, y) in number:
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 30, 0)
    pygame.image.save(screen, path.join("pictures", "config", \
                                           str(number_index) + ".png"))
    number_index += 1
