"""
File: weather_master.py
Name: Claire Lin
-----------------------
This program should implement a console program
that asks weather data from user to compute the
average, highest, lowest, cold days among the inputs.
Output format should match what is shown in the sample
run in the Assignment 2 Handout.

"""

EXIT = -100


def main():
    """
    To find the highest and lowest temperature, cold days and the average.
    """
    print('stanCode \"Weather Master 4.0\"!')

    # my friend told me the maximum and minimum variable can set like this.
    maximum = -100000000
    minimum = 100000000

    total = 0
    count = 0
    cold_day = 0

    while True:

        temperature = int(input('Next Temperature: (or '+str(EXIT) + ' to quit)? '))

        # To jump out from the program when no temperature were entered.
        if temperature == EXIT and count == 0:
            print('No temperatures were entered.')
            break

        # To exclude the temperature not exist.
        if temperature > 90 or temperature < -100:
            print('>>> The temperature \"'+str(temperature)+'\" not exist, so we exclude and stop it.')
            break

        if temperature == EXIT:
            break

        else:
            count += 1  # count the total days.

            if temperature < 16:
                cold_day += 1  # count the cold days which temperature below 16.

            total += temperature  # To plus all temperature.
            if temperature > maximum:
                maximum = temperature
            if temperature < minimum:
                minimum = temperature
            else:
                pass

    if count != 0:
        avg = total / count
        print("")
        print('Highest temperature = ' + str(maximum))
        print('Lowest temperature = ' + str(minimum))
        print('Average = '+str(avg))
        print(str(cold_day) + ' cold day(s)')
        # For checking
        # print(total)
        # print(count)

    """
    My note:
    This is the first try, when I debug I found the calculation logic is wrong.
    The first variable I type will disappear when it enter into the while loop. And the count of 
    total days would include the EXIT constant.
    """
    # if temperature == EXIT:
    #     print('No temperatures were entered.')
    #
    # else:
    #     while True:
    #         # if temperature < 16:
    #         #     cold_day += 1
    #
    #         temperature = int(input('Next Temperature: (or '+str(EXIT) + ' to quit)? '))
    #
    #         # count the total days.
    #         count += 1
    #
    #         if temperature == EXIT:
    #             break
    #
    #         total += temperature
    #         if temperature > maximum:
    #             maximum = temperature
    #         elif temperature < minimum:
    #             minimum = temperature
    #         else:
    #             pass
    #
    #     avg = total / count
    #     print('Highest temperature = ' + str(maximum))
    #     print('Lowest temperature = ' + str(minimum))
    #     print('Average = '+str(avg))
    #     print(str(cold_day) + ' cold day(s)')


###### DO NOT EDIT CODE BELOW THIS LINE ######

if __name__ == "__main__":
    main()
