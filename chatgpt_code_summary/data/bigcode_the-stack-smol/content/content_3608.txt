import kanu

while True:

    print('\n          Select one:')
    print('\t1 -> Solve a linear equation')
    print('\t2 -> Simplify any expression')
    print('\t3 -> Is this number a Perfect Square?')
    print('\t4 -> Get Prime Numbers')
    print('\t5 -> START A NUCLEAR WAR :)')
    print('\t6 -> Factor Integers')

    choice = input()

    if choice == '1':
        print('Enter the equation:', end=' ')

        try:
            print(kanu.solve_single_linear_equation(input()))
        except kanu.NonLinearEquationError:
            print('You entered a non-linear equation.')
    elif choice == '2':
        print('Enter the expression:', end=' ')
        print(kanu.all_together_now(input()))
    elif choice == '3':
        import math


        def is_perfect_square(y):
            sqrt_value = math.sqrt(y)
            return int(sqrt_value) ** 2 == y


        number = int(input('Enter a number: '))
        if is_perfect_square(number):
            print("It is a perfect square!")
        else:
            print("It is NOT a perfect square!")
    elif choice == '4':
        number = int(input("Input an integer:"))
        factors = []

        while number % 2 == 0:
            factors.append(2)
            number //= 2

        divisor = 3
        while number != 1 and divisor <= number:
            if number % divisor == 0:
                factors.append(divisor)
                number //= divisor
            else:
                divisor += 2
        print("The Prime Factors are: ")
        for i in range(len(factors)):
            print(factors[i], end=',')
    elif choice == '5':
        print('Executing "GETTING THE FOOTBALL"  ')
    if choice == '5':
        from tqdm import tqdm

        x = 1

        for i in tqdm(range(0, 1000000)):
            for x in range(0, 100):
                x *= 4
        print("DONE")

        print("HERE ARE THE NUCLEAR LAUNCH CODES...")

        print(" 56  58  10  62  11   1  25  29  55  62")

        print(" 5   8   1   9   6   7   4   3  10   20")

        print(" 41  16  18  50   9  51  48   5  37  30")

        print(" 40   3  34  61  59   2  39  46  28  47")

        print(" 38   7  42  26  63  45  17  27  60  21")

        print("Launch Nukes?")
        print("\t1 -> YES")
        print('\t2 -> NO')
        choice = input()
        if choice == '1':
            print('Please Wait...')
            from tqdm import tqdm

            x = 1

            for i in tqdm(range(0, 100000)):
                for x in range(0, 95):
                    x *= 4
            print('BYE BYE WORLD')
            input('press ENTER to continue')
        elif choice == '2':
            print('Maybe Another Day.')
            input('press ENTER to continue')
    elif choice == '6':
        import math

        number = int(input("Enter a number: "))
        factors = []

        for i in range(1, int(math.sqrt(number)) + 1):
            if number % i == 0:
                factors.append(i)
                factor_pair = number // i
                if factor_pair != i:
                    factors.append(factor_pair)
        factors.sort()
        print(factors)
