matrix_a = [[1,2,3], [4,5,6]]
result = [ [ element for element in t] for t in zip(*matrix_a)]
print(result)