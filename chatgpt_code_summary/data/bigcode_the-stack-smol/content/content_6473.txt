import itertools

def turn(face, dir):
    nface=face[:]
    if dir==1:
        nface=[0, face[2],face[6],face[3],face[1],face[5],face[4]]
    if dir==2:
        nface=[0, face[3],face[2],face[6],face[4],face[1],face[5]]
    if dir==3:
        nface=[0, face[4],face[1],face[3],face[6],face[5],face[2]]
    if dir==4:
        nface=[0, face[5],face[2],face[1],face[4],face[6],face[3]]
    return nface

def dfs(si, nowface):
    global link, visited
    result=True
    visited[si]=True

    if nowface[1] != si:
        return False
    
    for dir in range(1,5):
        if link[si][dir] and not visited[link[si][dir]]:
            face = turn(nowface, dir)
            result = result and dfs(link[si][dir], face)
    
    return result

x=[[0]*8]
for i in range(6):
    x.append([0] + list(map(int,input().split())) + [0])
x.append([0]*8)

link=[[None]*5 for i in range(10)]
for i in range(1, 7):
    for j in range(1, 7):
        if x[i][j]:
            if x[i-1][j]: link[x[i][j]][1]=x[i-1][j]
            if x[i+1][j]: link[x[i][j]][3]=x[i+1][j]
            if x[i][j-1]: link[x[i][j]][4]=x[i][j-1]
            if x[i][j+1]: link[x[i][j]][2]=x[i][j+1]

for i in itertools.permutations(map(int,'123456'), 6):
    face=list((0,)+i)
    visited=[0]*7
    if dfs(face[1], face) and sum(visited)>=6:
        print(face[6])
        exit(0)

print(0)