start_num = int(input())
end_num = int(input())

prime_nums = []


for num in range(start_num, end_num + 1):
    if num > 1:
        for i in range(2, num//2+1):
            if num % i == 0:
                break
        else:
            prime_nums.append(num)


print(' '.join(map(str, prime_nums)) + ' ')
print(f'The total number of prime numbers between {start_num} to {end_num} is {len(prime_nums)}')
