#link (https://neps.academy/problem/443)

voltas,placas= input().split()
result = int(voltas) * int(placas)
numbers = []
resultado = result * float(str(0) + str('.') + str(1))
for x in range(2,11):
    if int(resultado)==resultado:
        numbers.append(int(resultado))
    else:
        numbers.append(int(resultado)+1)
    resultado = result * float(str(0) + str('.') + str(x))
    
for x in numbers:
    print(int(x), end=' ')
