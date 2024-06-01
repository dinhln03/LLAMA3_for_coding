import pgzero
import pgzrun
import random

from pgzero.actor import Actor
__all__ = ["pgzrun", "pgzero"]

from pgzero.clock import clock

from pgzero.keyboard import keyboard
from pgzero.loaders import sounds

clouds = [Actor('cloud1', (200, 200)),
          Actor('cloud2', (400, 300)),
          Actor('cloud3', (600, 200)),
          Actor('cloud1', (800, 300))]

obstacles = [Actor('cactus', (random.randint(900, 1000), 495)),
             Actor('cactus', (random.randint(1200, 1500), 495)),
             Actor('cactus', (random.randint(1500, 2000), 495))]

player = Actor('p3_stand', (100, 484))

# 0 - game not started
# 1 - game just stared
# 2 - finished
game = 0
# frame that is currently running
frame = 0
# player movement speed and direction
jump = 0
# 0 - jump is available
# 1 - jump is forbidden
jump_blocked = 0
cloud_speed = 2
game_time = 0
# cactus movement speed
game_speed = 8
# 0 - game running
# 1 - game blocked
jump_unblocked = 0


def draw():
    global game
    screen.clear()
    screen.fill('#cff4f7')
    for i in range((screen.width // 70) + 1):
        screen.blit('grass', (i * 70, screen.height - 70))

    for cloud in clouds:
        cloud.draw()

    for obstacle in obstacles:
        obstacle.draw()

    screen.draw.text(
        align_text_time(game_time),
        midright=(screen.width - 50, 50),
        fontname="roboto_mono_bold",
        color="orange",
        fontsize=45
    )

    player.draw()

    if game == 0:
        screen.draw.text(
            "Wcisnij spacje",
            center=(screen.width / 2, screen.height / 2),
            color="orange",
            fontsize=60
        )

    if game == 2:
        screen.draw.text(
            "Koniec gry",
            center=(screen.width / 2, screen.height / 2),
            color="red",
            fontsize=60
        )
        screen.draw.text(
            "Wcisnij spacje aby zagrac jeszcze raz",
            center=(screen.width / 2, screen.height - 200),
            color="red",
            fontsize=30
        )


def update():
    global game
    global jump
    global jump_blocked
    global jump_unblocked

    if keyboard.SPACE and jump_unblocked == 0:
        if game == 0 or game == 2:
            jump_blocked = 1
            clock.schedule_unique(unblock_jump, 0.3)
            reset()
        game = 1
        if jump_blocked == 0:
            jump = -18
            jump_blocked = 1
            sounds.jingles_jump.play()
    animation()
    jump_fall()
    move_cloud()
    move_obstacle()
    check_collision()


# change difficulty level, increase game and clouds speed
def change_difficulty_level():
    global game_speed
    global cloud_speed
    if game_speed < 16:
        game_speed += 1
        cloud_speed += 1


# reset global variables
def reset():
    global frame
    global game
    global jump
    global jump_blocked
    global cloud_speed
    global game_speed
    global game_time
    if game == 2:
        frame = 0
        game = 0
        jump = 0
        jump_blocked = 1
        cloud_speed = 2
        game_speed = 8
        game_time = 0
        player.pos = (100, 484)
        clouds[0].pos = (200, 200)
        clouds[1].pos = (400, 300)
        clouds[2].pos = (600, 200)
        clouds[3].pos = (800, 300)
        obstacles[0].pos = (random.randint(900, 1000), 495)
        obstacles[1].pos = (random.randint(1200, 1500), 495)
        obstacles[2].pos = (random.randint(1500, 2000), 495)
        clock.unschedule(change_difficulty_level)
        # change difficulty level every 20s
        clock.schedule_interval(change_difficulty_level, 20)


def unblock_game():
    global jump_unblocked
    jump_unblocked = 0


# check collision with cactus
def check_collision():
    global game
    global jump_unblocked
    if game == 1:
        for i in obstacles:
            if player.collidepoint(i.x, i.y):
                game = 2
                sounds.jingles_end.play()
                jump_unblocked = 1
                # unblock game in 2 sec
                clock.schedule_unique(unblock_game, 2.0)


def move_obstacle():
    global game_speed
    global game
    if game == 1:
        for i in range(len(obstacles)):
            # decrease x for all obstacles about speed value
            obstacles[i].x -= game_speed
            # if obstacles is out of screen get random position
            if obstacles[i].x + 35 < 0:
                obstacles[i].x = random.randint(900, 1500)
                # if obstacles have the same position as other or is too close, move it about 400
                for j in range(0, len(obstacles)):
                    if j != i and abs(obstacles[i].x - obstacles[j].x < 300):
                        obstacles[i].x += 400


# triggered every 0.1s increasing game time about 1s
def measure_time():
    global game_time
    global game
    if game == 0:
        game_time = 0
    elif game == 1:
        game_time +=1


def align_text_time(time):
    text = "0" * (5 - len(str(time)))
    text += str(time)
    return text


def move_cloud():
    global cloud_speed
    global game
    if game == 1:
        # move clouds x pos about cloud speed
        for cloud in clouds:
            cloud.x -= cloud_speed
            # if cloud out of screen move it to right side
            if cloud.x + 64 < 0:
                cloud.x = screen.width + 32


def unblock_jump():
    global jump_blocked
    jump_blocked = 0


def jump_fall():
    global jump
    global frame
    if jump != 0:
        # block animation
        frame = 0
        player.y += jump
    # if player on the ground unblock
    if player.y >= 484:
        unblock_jump()
        jump = 0
    # if player jumped start falling
    if player.y <= 250:
        jump *= (-1)


# player animation
def animation():
    global frame
    if game == 1:
        if frame == 0:
            player.image = 'p3_walk01'
        if frame == 1:
            player.image = 'p3_walk02'
        if frame == 2:
            player.image = 'p3_walk03'
        if frame == 3:
            player.image = 'p3_walk04'
        if frame == 4:
            player.image = 'p3_walk05'
        if frame == 5:
            player.image = 'p3_walk06'
        if frame == 6:
            player.image = 'p3_walk07'
        if frame == 7:
            player.image = 'p3_walk08'
        if frame == 8:
            player.image = 'p3_walk09'
        if frame == 9:
            player.image = 'p3_walk10'
        if frame == 10:
            player.image = 'p3_walk11'
        frame += 1
        # result is 0 or less than 11
        frame %= 11


clock.schedule_interval(measure_time, 0.1)
clock.schedule_interval(change_difficulty_level, 20)
pgzrun.go()