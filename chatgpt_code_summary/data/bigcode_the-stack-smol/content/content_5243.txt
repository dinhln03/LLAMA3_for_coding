import random

### Advantage Logic ###
def advantage(rollfunc):
    roll1 = rollfunc
    roll2 = rollfunc
    if roll1 > roll2:
        return roll1
    else:
        return roll2
### Disadvantage Logic ###
def disadvantage(rollfunc):
    roll1 = rollfunc
    roll2 = rollfunc
    if roll1 < roll2:
        return roll1
    else:
        return roll2
### Die Rolls ###

def rolld4(sides:int=4):
    return random.randint(1, sides)

def rolld6(sides:int=6):
    return random.randint(1, sides)

def rolld8(sides:int=8):
    return random.randint(1, sides)

def rolld10(sides:int=10):
    return random.randint(1, sides)

def rolld12(sides:int=12):
    return random.randint(1, sides)

def rolld20(sides:int=20):
    return random.randint(1, sides)