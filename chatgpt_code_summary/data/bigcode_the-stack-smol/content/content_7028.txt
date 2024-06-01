# Interview Questions
"""
Given the following list of objects {user, loginTime, logoutTime}. What is the maximum number of concurrent users logged in at the same time?

	Input:
	[
	  {user: A, login: 1, logout: 3},
	  {user: B, login: 3, logout: 4},
	  {user: C, login: 1, logout: 2},
	  {user: D, login: 123123123, logout: 987987987},
	  {user: E, login: 1, logout: 3}
	]

	Output:
	3  
"""



datas = [
        {'user': 'A', 'login': 1, 'logout': 3},
        {'user': 'B', 'login': 3, 'logout': 4},
        {'user': 'C', 'login': 1, 'logout': 2},
        {'user': 'D', 'login': 123123123, 'logout': 987987987},
        {'user': 'E', 'login': 1, 'logout': 3}
]

from collections import Counter
def c(data):
    
    v = [[e['login'] for e in data],[e['logout'] for e in data]]
    t = [Counter(v[0]),Counter(v[1])]
    tmp = Counter()

    allt = list(set(v[0]+v[1]))
    allt.sort()


    ret = []
    cp = 0

    for e in allt:
        cp += t[0][e]
        cp -= t[1][e]

        ret.append(cp)

    return max(ret)


max = c(datas)
print(max)




