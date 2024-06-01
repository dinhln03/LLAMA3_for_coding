import networkx as nx
from operator import add
with open("input.txt","r") as f:
    char_map=[list(l.strip('\n')) for l in f.readlines()]

top_labels=char_map[:2]
top_labels=list(map(add,top_labels[0],top_labels[1]))
bottom_labels=char_map[-2:]
bottom_labels=list(map(add,bottom_labels[0],bottom_labels[1]))
char_map=char_map[2:-2]
for i,c in enumerate(char_map[0]):
    if c=='.':
        char_map[0][i]=top_labels[i]
        

for i,c in enumerate(char_map[-1]):
    if c=='.':
        char_map[-1][i]=bottom_labels[i]

left_labels=[]
right_labels=[]
for i in range(len(char_map)):
    right_labels.append((char_map[i].pop()+char_map[i].pop())[::-1])
    left_labels.append(char_map[i].pop(0)+char_map[i].pop(0))

for i in range(len(char_map)):
    if char_map[i][0]=='.':
        char_map[i][0]=left_labels[i]
    if char_map[i][-1]=='.':
        char_map[i][-1]=right_labels[i]

inner_begin_i=next( (i for i,x in enumerate(char_map) if ' ' in x))
inner_end_i=len(char_map)-next( (i for i,x in enumerate(reversed(char_map)) if ' ' in x))-1
inner_begin_j=char_map[inner_begin_i].index(' ')
inner_end_j=len(char_map[inner_end_i])-list(reversed(char_map[inner_end_i])).index(' ')-1

inner_top_labels=[]
inner_bottom_labels=[]
inner_left_labels=[]
inner_right_labels=[]
for i in range(inner_begin_i,inner_end_i):
    inner_left_labels.append(char_map[i][inner_begin_j]+char_map[i][inner_begin_j+1])
    inner_right_labels.append(char_map[i][inner_end_j-1]+char_map[i][inner_end_j])

for j in range(inner_begin_j,inner_end_j):
    inner_top_labels.append(char_map[inner_begin_i][j]+char_map[inner_begin_i+1][j])
    inner_bottom_labels.append(char_map[inner_end_i-1][j]+char_map[inner_end_i][j])

for i in range(inner_begin_i,inner_end_i):
    if char_map[i][inner_begin_j-1]=='.':
        char_map[i][inner_begin_j-1]=inner_left_labels[i-inner_begin_i]
    if char_map[i][inner_end_j+1]=='.':
        char_map[i][inner_end_j+1]=inner_right_labels[i-inner_begin_i]

for j in range(inner_begin_j,inner_end_j):
    if char_map[inner_begin_i-1][j]=='.':
        char_map[inner_begin_i-1][j]=inner_top_labels[j-inner_begin_j]
    if char_map[inner_end_i+1][j]=='.':
        char_map[inner_end_i+1][j]=inner_bottom_labels[j-inner_begin_j]

g=nx.Graph()
labels=dict()
for i in range(len(char_map)):
    for j in range(len(char_map[i])):
        if char_map[i][j]=='.':
            if char_map[i-1][j]!='#':#up
                g.add_edge((i,j),(i-1,j))
            if char_map[i+1][j]!='#':#down
                g.add_edge((i,j),(i+1,j))
            if char_map[i][j+1]!='#':#left
                g.add_edge((i,j),(i,j+1))
            if char_map[i][j-1]!='#':#right
                g.add_edge((i,j),(i,j-1))
        elif len(char_map[i][j])>1:#label
            if char_map[i][j] not in labels:
                labels[char_map[i][j]]=[(i,j)]
            else:
                labels[char_map[i][j]].append((i,j))
for v in labels.values():
    if len(v)==2:
        g.add_edge(v[0],v[1])
print(nx.shortest_path_length(g,labels["AA"][0],labels["ZZ"][0]))