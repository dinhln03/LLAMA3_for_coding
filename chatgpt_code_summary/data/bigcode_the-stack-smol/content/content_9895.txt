# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:29:57 2014

@author: garrett
"""

from user import User


def save_users(users, filename='output.csv'):
    '''Save users out to a .csv file

    Each row will represent a user UID, following by all the user's students
    (if the user has any)

    INPUT:
        > users: set of User objects
        > filename: filename to save .csv to.'''
    with open(filename, 'w') as file:
        for count, user in enumerate(users):
            file.write(str(user.get_uid()))
            for student in user.get_students():
                file.write(',' + str(student.get_uid()))
            file.write('\n')
            if count % 100 == 0:
                file.flush()
    return


def load_users(filename):
    '''Load users from a .csv file

    Each row will represent a user uid, following by all the user's student
    (if the user has any).  Note: the uid is not assumed to be an integer,
    so it read in as a string, which shouldn't matter anyway.

    TODO: we could probably speed this up by loading multiple lines at a time.

    INPUT:
        > filename: filename to read .csv from

    RETURN:
       > users: a set of User objects'''

    users = dict()

    # On first read, we create Users, on the following read, we save student
    # connections
    with open(filename, 'r') as file:
        for line in file:
            line = line.split('\n')[0]
            split_line = line.split(',')
            new_uid = _try_converting_to_int(split_line[0])
            new_user = User(new_uid)
            users.update({new_user.get_uid(): new_user})
    with open(filename, 'r') as file:
        for line in file:
            line = line.split('\n')[0]
            split_line = line.split(',')
            current_uid = _try_converting_to_int(split_line[0])
            for student_uid in split_line[1:]:
                student_uid = _try_converting_to_int(student_uid)
                users[current_uid].add_students(users[student_uid])

    return set(users.values())


def _try_converting_to_int(num):
    try:
        return int(num)
    except ValueError:
        return num
