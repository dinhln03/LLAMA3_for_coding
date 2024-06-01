class Point:
    counter = []

    def __init__(self, x=0, y=0):
        """Konstruktor punktu."""
        self.x = x
        self.y = y

    def update(self, n):
        self.counter.append(n)

p1 = Point(0,0)
p2 = Point(1,1)

p1.counter.append(1)
p2.counter.append(3)
p1.counter[0] = 2
print(p1.counter)
print(p2.counter)
p1.update(1)
print(p1.counter)
print(p2.counter)
