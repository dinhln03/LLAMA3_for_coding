#!/usr/bin/env python
import bottle
import os, json
from .utils import distance, neighbours, direction
from .defensive import find_my_tail, trouble, find_enemy_tail, eat_food, find_my_tail_emergency
from .snake import Snake
from .gameboard import GameBoard


SAFTEY = 0
SNAKE = 1
FOOD = 3
DANGER = 5


def move_response(move):
    assert move in ['up', 'down', 'left', 'right'], \
        "Move must be one of [up, down, left, right]"

    return bottle.HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({
            "move": move
            })
    )


def init(data):
    """
    Initialize grid and update cell values\n

    @param data -> Json response from bottle\n
    @return game_id -> Game id for debuggin purposes when displaying grid\n
    @return grid -> Grid with updated cell values\n
    @return food -> Sorted array of food by closest to charlie\n
    @return charlie -> My snake\n
    @return enemies -> Array of all enemy snakes\n
    @return check_food -> Secondary grid to look ahead when eating food
    """
    food = []
    enemies = []

    grid = GameBoard(data['board']['height'], data['board']['width'])
    check_food = GameBoard(data['board']['height'], data['board']['width'])

    charlie = Snake(data['you'])
  
    for i in data['board']['food']:
        food.append([i['x'], i['y']])
        grid.set_cell([i['x'], i['y']], FOOD)
        check_food.set_cell([i['x'], i['y']], FOOD)
    
    for snake in data['board']['snakes']:
        snake = Snake(snake)
        for coord in snake.coords:
            grid.set_cell(coord, SNAKE)
            check_food.set_cell(coord, SNAKE)
        if snake.health < 100 and snake.length > 2 and data['turn'] >= 3:
            grid.set_cell(snake.tail, SAFTEY)
            check_food.set_cell(snake.tail, SAFTEY)
        if snake.id != charlie.id:
            for neighbour in neighbours(snake.head, grid, 0, snake.coords, [1]):
                if snake.length >= charlie.length:
                    grid.set_cell(neighbour, DANGER)
                    check_food.set_cell(neighbour, DANGER)
            enemies.append(snake)

    food = sorted(food, key = lambda p: distance(p, charlie.head))

    game_id = data['game']['id']
    # print("turn is {}".format(data['turn']))
    return game_id, grid, food, charlie, enemies, check_food


@bottle.post('/ping')
def ping():
    return bottle.HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({})
    )


@bottle.post('/start')
def start():
    return bottle.HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({
            "color": '#002080',
            'headType': 'pixel',
            'tailType': 'pixel'
        })
    )


@bottle.post('/move')
def move():
    data = bottle.request.json

    game_id, grid, food, charlie, enemies, check_food = init(data)
    
    # grid.display_game(game_id)

    if len(enemies) > 2 or charlie.length <= 25 or charlie.health <= 60:
        path = eat_food(charlie, grid, food, check_food)
        if path:
            # print('eat path {}'.format(path))
            return move_response(direction(path[0], path[1]))

    if charlie.length >= 3:
        path = find_my_tail(charlie, grid)
        if path:
            # print('find my tail path {}'.format(path))
            return move_response(direction(path[0], path[1]))

    if not path:
        path = find_enemy_tail(charlie, enemies, grid)
        if path:
            # print('find enemy tail path {}'.format(path))
            return move_response(direction(path[0], path[1]))

    # # if our length is greater than threshold and no other path was available
    if charlie.length >= 3:
        path = find_my_tail_emergency(charlie, grid)
        if path:
            # print('find my tail emergency path {}'.format(path))
            return move_response(direction(path[0], path[1]))

    # Choose a random free space if no available enemy tail
    if not path:
        path = trouble(charlie, grid)
        if path:
            # print('trouble path {}'.format(path))
            return move_response(direction(path[0], path[1]))


@bottle.post('/end')
def end():
	return bottle.HTTPResponse(
        status=200,
        headers={
            "Content-Type": "application/json"
        },
        body=json.dumps({})
    )


application = bottle.default_app()
if __name__ == '__main__':
	bottle.run(application, host=os.getenv('IP', '0.0.0.0'), port=os.getenv('PORT', '8080'), quiet = True)