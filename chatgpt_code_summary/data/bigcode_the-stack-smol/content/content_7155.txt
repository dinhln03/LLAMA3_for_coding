def solution(S, T):
    #대소문자 변형
    S = S.lower()
    T = T.lower()

    dic = {}
    for i in S:
        if dic.get(i):
            dic[i] += 1
        else:
            dic[i] = 1
    for i in T:
        if dic.get(i):
            dic[i] -= 1
        else:
            dic[i] = -1

    for i in dic.values():
        if i != 0:
            return False
    return True

S = "DOcING"
T = "CODING"

print(solution(S, T))