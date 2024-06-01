# IDLE (Python 3.8.0)

# module_for_lists_of_terms


def termal_generator(lict):
    length_of_termal_generator = 16
    padding = length_of_termal_generator - len(lict)
    count = padding
    while count != 0:
        lict.append([''])
        count = count - 1
    termal_lict = []
    for first_inner in lict[0]:
        for second_inner in lict[1]:
            for third_inner in lict[2]:
                for fourth_inner in lict[3]:
                    for fifth_inner in lict[4]:
                        for sixth_inner in lict[5]:
                            for seventh_inner in lict[6]:
                                for eighth_inner in lict[7]:
                                    for ninth_inner in lict[8]:
                                        for tenth_inner in lict[9]:
                                            for eleventh_inner in lict[10]:
                                                for twelfth_inner in lict[11]:
                                                    for thirteenth_inner in lict[12]:
                                                        for fourteenth_inner in lict [13]:
                                                            for fifteenth_inner in lict [14]:
                                                                for sixteenth_inner in lict[15]:
                                                                    term = (
                                                                      first_inner + second_inner +
                                                                      third_inner + fourth_inner +
                                                                      fifth_inner + sixth_inner +
                                                                      seventh_inner + eighth_inner +
                                                                      ninth_inner + tenth_inner +
                                                                      eleventh_inner + twelfth_inner +
                                                                      thirteenth_inner + fourteenth_inner +
                                                                      fifteenth_inner + sixteenth_inner
                                                                    )
                                                                    termal_lict.append(term)
    return termal_lict

def user_input_handling_function_second(dictionary):
    print()
    user_input = input('Enter: ')
    print()
    good_to_go = 'no'
    errors = []
    lict = []
    while good_to_go == 'no':
        for key in dictionary:
            lict.append(key)
        for element in user_input:
            if element not in lict:
                print('The form can only contain a combination of the characters that represent the lists of characters.')
                errors.append('yes')
                break
        if len(user_input) < 2:
            print('The form is too short. It can\'t be less than two-characters long.')
            errors.append('yes')
        if len(user_input) > 8:
            print('The form is too long. It can\'t be more than eight-characters long.')
            errors.append('yes')
        if 'yes' in errors:
            good_to_go = 'no'
            errors = []
            print()
            user_input = input('Re-enter: ')
            print()
        else:
            good_to_go = 'yes'
    return user_input

def user_input_handling_function_third(): 
    print()
    user_input = input('Enter: ')
    print()
    good_to_go = 'no'
    errors = []
    yes_or_no = ['yes', 'no']
    while good_to_go == 'no':
        if user_input not in yes_or_no:
            print('You have to answer yes or no.')
            errors.append('yes')
        if 'yes' in errors:
            good_to_go = 'no'
            errors = []
            print()
            user_input = input('Re-enter: ')
            print()
        else:
            good_to_go = 'yes'
    return user_input
    
def user_input_handling_function_fourth(dictionary):
    print()
    user_input = input('Enter: ')
    print()
    good_to_go = 'no'
    errors = []
    while good_to_go == 'no':
        if user_input not in dictionary:
            print('The form you entered does not match one of the forms in your termal_dictionary. Each form in your')
            print('termal_dictionary is a name (key) that has an associated definition (value) that is a list of terms')
            print('that all have the same form as the name (key).')
            errors.append('yes')
        if 'yes' in errors:
            good_to_go = 'no'
            errors = []
            print()
            user_input = input('Re-enter: ')
            print()
        else:
            good_to_go = 'yes'
    return user_input

def user_input_handling_function_eighth():
    print()
    user_input = input('Enter: ')
    print()
    good_to_go = 'no'
    errors = []
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
    while good_to_go == 'no':
        if user_input == 'None':
            user_input = None
            return user_input
        else:
            for inner in user_input:
                if inner not in digits:
                    print('The number must be an integer that consists of digits. For example: 1, -2, etc. or the keyword:')
                    print('None.')
                    errors.append('yes')
                    break
        if 'yes' in errors:
            good_to_go = 'no'
            errors = []
            print()
            user_input = input('Re-enter: ')
            print()
        else:
            good_to_go = 'yes'
    return int(user_input)

def user_input_handling_function_ninth():
    ''' a parser '''
    print()
    user_input = input('Enter: ')
    print()
    term = ''
    lict = []
    for element in user_input:
        if element != ' ':
            term = term + element
        else:
            lict.append(term)
            term = ''
    lict.append(term) # because term might not be empty....
    return lict

def user_input_handling_function_tenth(dictionary):
    ''' a dictionary checker '''
    user_input = user_input_handling_function_ninth()
    good_to_go = 'no'
    errors = []
    while good_to_go == 'no':
        string = ''
        lict = []
        for element in user_input:
            string = string + element
        for key in dictionary:
            for element in dictionary[key]:
                lict.append(element)
        for element in string:
            if element not in lict:
                print('One of your unwanted characters or combination of characters does not match the characters you')
                print('entered earlier.')
                errors.append('yes')
                break
        if 'yes' in errors:
            print()
            user_input = input('Re-enter: ')
            print()
            good_to_go = 'no'
            errors = []
        else:
            good_to_go = 'yes'
    return user_input
    
def print_vertical_lict(lict):
    for element in lict:
        print(element)

def print_horizontal_lict(lict):
    string = ''
    for element in lict:
        string = string + str(element) + ', '
    print(string)
    print()

def write_vertical_lict(file_name, lict): # <--
    file = open(file_name, 'w')
    for element in lict:
        element = str(element) + '\n'
        file.write(element)
    file.close()

def write_horizontal_lict(file_name, lict):
    if '.txt' not in file_name:
        file_name = file_name + '.txt'
    row = ''
    for index in range(len(lict)):
        lict[index] = str(lict[index]) + ', '
        if len(row + lict[index]) > 100:
            lict[index - 1] = lict[index - 1] + '\n'
            row = lict[index]
        else:
            row = row + lict[index]
    file = open(file_name, 'w')
    for term in lict:
        file.write(term)
    file.close()

