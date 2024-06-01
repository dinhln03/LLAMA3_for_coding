class Person(object):
    """ Class Person for testing python.

    Following packages need to be installed:
     - requests

    :param name: person's name, string
    :param age: person's age, integer
    :param phone: person's phone, string
    :rtype: object
    """
    def __init__(self, name, age, phone):
        self.name = name
        self.age = age
        self.phone = phone

    def print(self):
        """ Method prints person's data.

        :return: None
        """
        print("Name: {}, age: {}, phone: {}".format(self.name, self.age, self.phone))

    def set_name(self, name):
        """ Method saves a new name for the person.

        :param name: new name for the person, string
        :return: None
        """
        self.name = name

    def get_name(self):
        """ Method returns the name of the person.

        :return: name, string
        """
        return self.name

    def set_age(self, age):
        """ Method saves a new age for the person.

        :param age: new age for the person, integer
        :return: None
        """
        if type(age) != int:
            print("not valid age {}".format(age))
            return

        if age >= 0:
            self.age = age
        else:
            print("not valid age {}".format(age))

    def get_age(self):
        """ Method returns the age of the person.

        :return: age, integer
        """
        return self.age

    def set_phone(self, phone):
        """ Method saves a new phone for the person.

        :param phone: new phone for the person, string
        :return: None
        """
        self.phone = phone

    def get_phone(self):
        """ Method returns the phone of the person.

        :return: phone, string
        """
        return self.phone


class Employee(Person):
    """ Class Employee for testing python.

    :param name: person's name, string
    :param age: person's age, integer
    :param phone: person's phone, string
    :param phone: person's title, string
    :param phone: person's salary, string
    :param phone: person's location, string
    :rtype: object
    """
    def __init__(self, name, age, phone, title, salary, location):
        super().__init__(name, age, phone)
        self.title = title
        self.salary = salary
        self.location = location

    def get_title(self):
        """ Method returns the title of the person.

        :return: title, string
        """
        return self.title

    def set_title(self, title):
        """ Method saves a new title for the person.

        :param title: new title for the person, string
        :return: None
        """
        self.title = title

    def get_salary(self):
        """ Method returns the salary of the person.

        :return: salary, string
        """
        return self.salary

    def set_salary(self, salary):
        """ Method saves a new salary for the person.

        :param salary: new salary for the person, string
        :return: None
        """
        if salary >= 0:
            self.salary = salary

    def get_location(self):
        """ Method returns the location of the person.

        :return: location, string
        """
        return self.location

    def set_location(self, location):
        """ Method saves a new location for the person.

        :param location: new location for the person, string
        :return: None
        """
        self.location = location

    def print_businesscard(self):
        """ Method prints a business card information.

        :return: None
        """
        print(" Name: {}\n Title: {}\n Phone: {}".format(self.name, self.title, self.phone))
