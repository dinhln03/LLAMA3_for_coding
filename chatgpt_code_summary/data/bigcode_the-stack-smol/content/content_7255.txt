# -*- coding: utf-8 -*-

check_state = 0

d = {}

p = []
e = []
m = []

n = int(input())

for _ in range(n):

    ln = input().split()

    d[ln[0]] = (int(ln[1]), int(ln[2]), int(ln[3]))

    p.append(int(ln[1]))
    e.append(int(ln[2]))
    m.append(int(ln[3]))

while True:

    if check_state == 0:

        if p.count(max(p)) == 1:
            
            for k in d:

                if d[k][0] == max(p):

                    print(k)
                    break
            
            break
        else:

            del_list = []

            for k in d:

                if d[k][0] != max(p):

                    p.remove(d[k][0])
                    e.remove(d[k][1])
                    m.remove(d[k][2])

                    del_list.append(k)
            
            for k in del_list:

                del d[k]
    
    if check_state == 1:

        if e.count(max(e)) == 1:

            for k in d:

                if d[k][1] == max(e):

                    print(k)
                    break
            
            break
        else:

            del_list = []

            for k in d:

                if d[k][1] != max(e):

                    p.remove(d[k][0])
                    e.remove(d[k][1])
                    m.remove(d[k][2])

                    del_list.append(k)
            
            for k in del_list:

                del d[k]

    if check_state == 2:

        if m.count(min(m)) == 1:

            for k in d:

                if d[k][2] == min(m):

                    print(k)
                    break
            
            break
        else:

            del_list = []

            for k in d:

                if d[k][2] != min(m):

                    p.remove(d[k][0])
                    e.remove(d[k][1])
                    m.remove(d[k][2])

                    del_list.append(k)
            
            for k in del_list:

                del d[k]
            
            # Ordem lexicográfica é a mesma coisa que afabética nesse caso
            keys = sorted(d.keys())
            print(keys[0])

            break

    check_state += 1