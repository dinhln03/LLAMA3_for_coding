import numpy as np
import math
from arena import Arena
from agent import HAgent, AAgent
import random
# np.random.seed(1234)


# place the humans on the arena
def place_soldiers(n, arena, agents):
    x = 0
    y = 0

    for i in range(n):
        agents[i + 1] = HAgent([x, y])
        arena.arena[x, y] = 1
        y += 2

    return arena, agents


# place the alien agents on the arena
def place_targets(n, arena, targets, pos=None):
    if pos is not None:
        orig_pos = pos
        for i in range(n):
            targets[i + 1] = AAgent(pos[i])
            arena.arena[pos[i][0], pos[i][1]] = 2
    else:
        orig_pos = []
        for i in range(n):
            while True:
                x = np.rint(np.array([(arena.size - 1) * np.random.rand(1),
                                      (arena.size - 1) * np.random.rand(1)]))
                if x[0] > 7 or x[1] > 7:
                    break
            x = [int(i) for i in x]
            # x = [19, 19]
            targets[i + 1] = AAgent(x)
            arena.arena[x[0], x[1]] = 2
            orig_pos.append([x[0], x[1]])
    return arena, targets, orig_pos


# adds half-cover tiles in random locations in the arena
# At most n cover tiles added, though potentially fewer
def place_half_cover(n, arena):
    for i in range(n):
        x = np.random.randint(0, (arena.size - 1))
        y = np.random.randint(0, (arena.size - 1))
        if arena.arena[x, y] == 0:
            arena.arena[x, y] = 3
    return arena


# movement for agents
def move(agent, arena, loc):
    # Check that agent has movement, if not, do nothing
    if agent.moves <= 0:
        # print('unsuccessful move')
        return agent, arena
    # Check if in movement range
    elif abs((loc[0] - agent.pos[0]) + (loc[1] - agent.pos[1])) <= agent.move_range:
        # print('successful move')
        # update the arena matrix
        arena.arena[agent.pos[0], agent.pos[1]] = 0
        arena.arena[loc[0], loc[1]] = 1
        # update agent location, number of moves
        agent.moves -= 1
        agent.pos = loc
        arena.time += 1
        return agent, arena
    # if not in movement range, do nothing
    else:
        # print('unsuccessful move')
        return agent, arena


# reload action
def reload(agent):
    if agent.moves > 0:
        agent.moves -= 1
        agent.ammo = 5
    return agent


def fire(agent, arena, target):
    # for the moment, assume anything can be fired on
    # set firing agent's moves to zero
    agent.moves = 0
    agent.ammo -= 1
    cover = 0
    # check if target is in (half) cover
    if agent.pos[0] + 1 > target.pos[0]:
        if arena.arena[target.pos[0] - 1, target.pos[1]] == 3:
            cover = 20
    if agent.pos[0] - 1 < target.pos[0]:
        if arena.arena[target.pos[0] + 1, target.pos[1]] == 3:
            cover = 20
    if agent.pos[1] + 1 > target.pos[1]:
        if arena.arena[target.pos[0], target.pos[1] - 1] == 3:
            cover = 20
    if agent.pos[1] - 1 < target.pos[1]:
        if arena.arena[target.pos[0], target.pos[1] + 1] == 3:
            cover = 20
    # for distance equation, see
    # https://www.ufopaedia.org/index.php/Chance_to_Hit_(EU2012)

    diff = [agent.pos[0] - target.pos[0], agent.pos[1] - target.pos[1]]
    distance_chance = 42 - 4.5 * (np.linalg.norm(diff))
    # Hit chance is base aim, less cover, plus distance modifier
    to_hit = agent.aim - cover + distance_chance
    if np.random.randint(100) >= to_hit:
        # miss, so no change
        arena.time += 1
        return agent, arena, target
    else:
        flanking = 0
        crit_modifier = 1
        # check if critical
        if cover == 0:
            flanking = 50
        crit_chance = agent.base_crit + flanking
        # crit modifier in xcom is 1.5x damage
        if np.random.randint(100) < crit_chance:
            crit_modifier = 1.5
        # slight random variance from base damage, +1 to -1
        damage = math.floor(crit_modifier * (np.random.randint(-1, 2) + agent.damage))
        # apply damage and return
        target.health -= damage
        # check if damage causes death
        arena, target = check_death_enemy(arena, target)
        arena.time += 1
        return agent, arena, target


# check to see if character is dead, update arena information if so
def check_death_enemy(arena, target):
    if target.health <= 0:
        target.moves = 0
        arena.arena[target.pos] = 0
        arena.targets -= 1
        if arena.targets <= 0:
            arena.targets = 0
    return arena, target


# refresh movement for non-dead characters
def new_turn(arena, agents, targets):
    for i in agents:
        if i.health > 0:
            i.moves = 2
    for j in targets:
        if j.health > 0:
            j.moves = 2

    return arena, agents, targets


# get a valid move
def get_valid_move(agent):
    x_old = agent.pos[0]
    y_old = agent.pos[1]
    # print(x_old, y_old)
    x = int(random.randint(x_old - 3, x_old + 3))
    y = int(random.randint(y_old - 3, y_old + 3))
    if x < 0:
        x = x * -1
    if y < 0:
        y = y * -1
    if x > 19:
        x = 19
    if y > 19:
        y = 19
    # print(x, y)
    return x, y
