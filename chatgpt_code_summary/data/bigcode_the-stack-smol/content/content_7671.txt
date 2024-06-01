"""
[1,0,1,0,1] -> correct
[0,0,1,0] -> return 1
[1,1,0,0,1] -> return 2
"""

def solution2(S):
    ans1, ans2 = 0, 0
    check1, check2 = 0, 1

    for i in S:
        if i != check1:
            ans1 +=1
        if i != check2:
            ans2 += 1
        check1 = 0 if check1 == 1 else 1
        check2 = 0 if check2 == 1 else 1
        assert(check1 != check2)

    print("ans1 : {}, ans2 : {}".format(ans1, ans2))
    return min(ans1, ans2)

def solution(S):
    ans1, ans2 = 0, 0
    S1 = S.copy()
    S2 = S.copy()
    leng = len(S)
    
    if leng == 1:
        return 0
    if leng == 2:
        if S[0] != S[1]:
            return 0
        else:
            return 1
    
#     Forward
    for idx in range(leng):
        if idx == 0 or idx == leng-1:
            continue
#         [0,0,0] or [1,1,1]    
        if S1[idx] == S1[idx-1] and S1[idx] == S1[idx+1]:
            if S1[idx] == 1:
                S1[idx] = 0
            else:
                S1[idx] = 1
            ans1 += 1
#         [0, 0, 1]
        if S1[idx] == S1[idx-1] and S1[idx] != S1[idx+1]:
            if S1[idx-1] == 1:
                S1[idx-1] = 0
            else:
                S1[idx-1] = 1
            ans1 += 1
#         [1,0,0]
        if S1[idx] != S1[idx-1] and S1[idx] == S1[idx+1]:
            if S1[idx+1] == 1:
                S1[idx+1] = 0
            else:
                S1[idx+1] = 1
            ans1 += 1
            
#     backwards      
    for idx in range(leng-1,-1,-1):
        if idx == 0 or idx == leng-1:
            continue
#         [0,0,0] or [1,1,1]    back
        if S2[idx] == S2[idx-1] and S2[idx] == S2[idx+1]:
            if S2[idx] == 1:
                S2[idx] = 0
            else:
                S2[idx] = 1
            ans2 += 1
#         [0, 0, 1]
        if S2[idx] == S2[idx-1] and S2[idx] != S2[idx+1]:
            if S2[idx-1] == 1:
                S2[idx-1] = 0
            else:
                S2[idx-1] = 1
            ans2 += 1
#         [1,0,0]
        if S2[idx] != S2[idx-1] and S2[idx] == S2[idx+1]:
            if S2[idx+1] == 1:
                S2[idx+1] = 0
            else:
                S2[idx+1] = 1
            ans2 += 1
            
    return min(ans1, ans2)


# print(solution([0,1,0,1,0,0,0,0,0,1]))
# print(solution([1,0,0,0,0,1,0]))
# print(solution([1,0,0]))
# print(solution([0,0,1]))
# print(solution([1,0,1]))
# print(solution([0,1,0]))

# print(solution2([0,1,0,1,0,0,0,0,0,1]))
print(solution([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))
print(solution2([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]))

# print(solution2([1,0,0]))
# print(solution2([0,0,1]))
# print(solution2([1,0,1]))
# print(solution2([0,1,0]))