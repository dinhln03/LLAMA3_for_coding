lengths = {0: 0, 1: 1}

def sequenceLength(n: int) -> int:
    global lengths
    if n not in lengths:
        if n % 2 == 0:
            lengths[n] = sequenceLength(n//2) + 1
        else:
            lengths[n] = sequenceLength(3 * n + 1) + 1  
    return lengths[n]

def solution(n: int = 1000000) -> int:
    result = 0
    maxLength = 0

    for i in range(n):
        counter = sequenceLength(i)
        if counter > maxLength:
            result = i
            maxLength = counter
    return result

print(solution())