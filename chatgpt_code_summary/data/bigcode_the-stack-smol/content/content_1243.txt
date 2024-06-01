from credentials import credentials
import unittest
import pyperclip 

class TestUser(unittest.TestCase):
    '''
    Test that defines test cases for the User class
    Args:
        unitest.Testcase: Testcase that helps in creating test cases for class User.
    '''

    def setUp(self):
        '''
        Set up method to run before each test case
        '''
        self.new_user = credentials("Paul", "123")

    def test__init__(self):
        '''
        test__init__ test case to test if the object is initialized properly
        '''

        self.assertEqual(self.new_user.user_name, "Paul")
        self.assertEqual(self.new_user.password, "123")

    def test__save_user(self):
        '''
        test to see if the user is saved
        '''
        self.new_credentials.save_credentials()
        self.assertEqual(len(credentials.user_list), 1)

    if __name__ == "__main__":
       unittest.main()
