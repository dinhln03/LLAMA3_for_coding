'''

Advent of Code - 2019

    --- Day 2: 1202 Program Alarm ---

'''

from utils import *
from intcode import IntcodeRunner, HaltExecution

def parse_input(day):
    return day_input(day, integers)[0]

def part1(program, noun=12, verb=2):
    runner = IntcodeRunner(program)
    runner.set_mem(1, noun)
    runner.set_mem(2, verb)

    while True: 
        try:
            next(runner.run())
        except HaltExecution:
            break

    return runner.get_mem(0)

def part2(program, target=19690720):
    runner = IntcodeRunner(program)
    for noun in range(100, -1, -1):
        for verb in range(100):
            runner.set_mem(1, noun)
            runner.set_mem(2, verb)
            while True:
                try:
                    next(runner.run())
                except HaltExecution:
                    break

            if runner.get_mem(0) == target:
                return 100*noun+verb

            runner.reset()
            
if __name__ == '__main__':
    data = parse_input('02')

    print(f'Part One: {part1(data)}')
    print(f'Part Two: {part2(data)}')
