enum = 0
enum1 = 0
enum2 = 0
prob = 0
p1 = 0
p2 = 0
parity = 0

for z1 in range(1, 6):
    for y1 in range(z1+1, 7):
        for z2 in range(1, z1+1):
            for y2 in range(z2+1, y1+1):
                """ for y2 in range(1, y1):
                    for z2 in range(y2, z1+1):
                        for z3 in range(1, z2+1):
                            if y1 == y2:
                                enum1 = 1
                            elif y1 > y2:
                                enum1 = 2
                            else:
                                enum1 = 0
                            p1 = enum1/36

                            if z1 == z2 == z3:
                                enum2 = 1
                            elif z1 != z2 != z3:
                                enum2 = 6
                            else:
                                enum2 = 3
                            p2 = enum2/216

                            enum += enum1 * enum2
                            prob += p1 * p2 """
                # print(y1, z1, y2, z2)
                if z1 == z2:
                    enum1 = 1
                elif z1 > z2:
                    enum1 = 2
                else:
                    enum1 = 0
                p1 = enum1 / 36

                if y1 == y2:
                    enum2 = 1
                elif y1 > y2:
                    enum2 = 2
                else:
                    enum2 = 0
                p2 = enum2 / 36
            
                enum += enum1 * enum2
                prob += p1 * p2

print(enum, prob)