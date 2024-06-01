INPUTPATH = "input.txt"
#INPUTPATH = "input-test.txt"
with open(INPUTPATH) as ifile:
	raw = ifile.read()
from typing import Tuple
def line_to_pos(line: str) -> Tuple[int, ...]:
	filtered = "".join(c for c in line if c.isdigit() or c in {"-", ","})
	return tuple(map(int, filtered.split(",")))
starts = tuple(zip(*map(line_to_pos, raw.strip().split("\n"))))

from itertools import combinations
from typing import List, Iterable
class Axis:
	poss: List[int]
	vels: List[int]
	def __init__(self, start_poss: Iterable[int]) -> None:
		self.poss = list(start_poss)
		self.vels = [0] * len(self.poss)
	def __eq__(self, other) -> bool:
		return self.poss == other.poss and self.vels == other.vels
	def step(self) -> None:
		for i, j in combinations(range(len(self.poss)), 2):
			a, b = self.poss[i], self.poss[j]
			diff = 1 if a < b else -1 if a > b else 0
			self.vels[i] += diff
			self.vels[j] -= diff
		for i, vel in enumerate(self.vels):
			self.poss[i] += vel

system = tuple(map(Axis, starts))
for axis in system:
	for _ in range(1000):
		axis.step()
pos_by_moon = zip(*(axis.poss for axis in system))
vel_by_moon = zip(*(axis.vels for axis in system))
print(sum(
	sum(map(abs, pos)) * sum(map(abs, vel))
	for pos, vel in zip(pos_by_moon, vel_by_moon)
))

def cycle_period(start_poss: Iterable[int]) -> int:
	tort = Axis(start_poss)  # Get some rest, buddy. :3
	hare = Axis(tort.poss)  # Up for a run? >:3c
	hare.step()
	steps = 1
	while hare != tort:
		hare.step()
		steps += 1
	return steps
from math import lcm
print(lcm(*map(cycle_period, starts)))
