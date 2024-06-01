# Advent of Code 2021 - Day: 24
# Imports (Always imports data based on the folder and file name)
from aocd import data, submit

def solve(lines):
	# We need to simply find all the pairs of numbers, i.e. the numbers on lines 6 and 16 and store them.
	pairs = [(int(lines[i * 18 + 5][6:]), int(lines[i * 18 + 15][6:])) for i in range(14)]
	# Once getting the pairs we will need a stack and a map to store the pairs, as well constraints.
	stack = []
	constraints = {}
	# Enumerate helps because we can get the index of the pair at the same time.
	for i, (a, b) in enumerate(pairs):
		# If (line 6) is positive we need to add line 16 and index to stack, else pop the last element from the stack and add it to constraints.
		if a > 0:
			stack.append((i, b))
		else:
			k, bk = stack.pop()
			constraints[i] = (k, bk + a)

	# At this point the constraints are stored at the relevant index for which they affect and can be used to find the minimum or maximum element at that index in the answer.
	max_ans = {}
	min_ans = {}
	for i, (k, d) in constraints.items():
		max_ans[i] = min(9, 9 + d)
		max_ans[k] = min(9, 9 - d)
		min_ans[i] = max(1, 1 + d)
		min_ans[k] = max(1, 1 - d)

	p1 = "".join(str(max_ans[i]) for i in range(14))
	p2 = "".join(str(min_ans[i]) for i in range(14))

	print("Star 1:", p1)
	print("Star 2:", p2)

	submit(p1, part="a", day=24, year=2021)
	submit(p2, part="b", day=24, year=2021)

# Solution
def main():
	solve(data.splitlines())

# Call the main function.
if __name__ == '__main__':
	main()