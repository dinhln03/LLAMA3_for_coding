from combat import Combat, RecursiveCombat

with open("test_input.txt") as f:
    game = Combat.parse_from_file(f)
    f.seek(0)
    recursive_game = RecursiveCombat.parse_from_file(f)

# 1st round
round_1 = game[1]

assert round_1.player_1==[2, 6, 3, 1, 9, 5]
assert round_1.player_2==[8, 4, 7, 10]

# 28th round
round_28 = game[27]

assert round_28.player_1==[4, 1]
assert round_28.player_2==[9, 7, 3, 2, 10, 6, 8, 5]

# error if checking score/winner before end
caught_error = False
try:
    game[12].score
except ValueError:
    caught_error = True
assert caught_error

caught_error = False
try:
    game[8].score
except ValueError:
    caught_error = True
assert caught_error

# end
end = game.play()

assert end.player_1==[]
assert end.player_2==[3, 2, 10, 6, 8, 5, 9, 4, 7, 1]

assert end.score==306

assert end.winner==1

# recursive game
end = recursive_game.play()

assert end.score==291
