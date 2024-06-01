import math

# 1
area_of_circle = lambda r: math.pi * r ** 2
print(area_of_circle(10))

# 2
calculation = lambda x, y: ((x + y), (x - y))
print(calculation(4, 2))


# 3
def product(n):
    if n == 1:
        return 1
    else:
        return n * product(n - 1)


print(product(5))

# 4
time = lambda milli: (
    round(milli / (1000 * 60 * 60 * 24)), round(milli / (1000 * 60 * 60)), round(milli / (1000 * 60)),
    round(milli / 1000))
print(time(10000000))

# 5
showSalary = lambda name, salary=5000: (name, salary)
print(showSalary("A", 1000))
print(showSalary("B", 2000))
print(showSalary("C"))

# 6
diff = lambda x, y: x - y
print(diff(10, 12))

# 7
printer = lambda x, y, z: (x, str(x), y, str(y), z, str(z))
print(printer(True, 22.25, 'yes'))

# 8
data = [
    ('Alpha Centauri A', 4.3, 0.26, 1.56),
    ('Alpha Centauri B', 4.3, 0.077, 0.45),
    ('Alpha Centauri C', 4.2, 0.00001, 0.00006),
    ("Barnard's Star", 6.0, 0.00004, 0.0005),
    ('Wolf 359', 7.7, 0.000001, 0.00002),
    ('BD +36 degrees 2147', 8.2, 0.0003, 0.006),
    ('Luyten 726-8 A', 8.4, 0.000003, 0.00006),
    ('Luyten 726-8 B', 8.4, 0.000002, 0.00004),
    ('Sirius A', 8.6, 1.00, 23.6),
    ('Sirius B', 8.6, 0.001, 0.003),
    ('Ross 154', 9.4, 0.00002, 0.0005),
]
# data.sort()
print(data)
print(sorted(data))