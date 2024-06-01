import pygame

# TODO: make these configurable

c_UP = pygame.K_UP
c_DOWN = pygame.K_DOWN
c_LEFT = pygame.K_LEFT
c_RIGHT = pygame.K_RIGHT

c_PREV = pygame.K_LEFTBRACKET
c_NEXT = pygame.K_RIGHTBRACKET
c_START = pygame.K_RETURN

c_1 = pygame.K_1
c_2 = pygame.K_2
c_3 = pygame.K_3
c_4 = pygame.K_4
c_5 = pygame.K_5
c_6 = pygame.K_6
c_7 = pygame.K_7
c_8 = pygame.K_8
c_9 = pygame.K_9
c_0 = pygame.K_0
c_POINT = pygame.K_PERIOD
c_DEL = pygame.K_BACKSPACE

c_X = pygame.K_a
c_A = pygame.K_x
c_B = pygame.K_z
c_Y = pygame.K_LSHIFT
c_L = pygame.K_q
c_R = pygame.K_w

def isDown(code):
    return pygame.key.get_pressed()[code]

def isUp(code):
    return not isDown(code)
