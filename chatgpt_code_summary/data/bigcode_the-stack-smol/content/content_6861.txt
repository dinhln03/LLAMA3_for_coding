class Student:
    def __init__(self):
        self.surname = None
        self.name = None
        self.patronymic = None
        self.age = 19
        self.birthday = None
        self.group = None


class Teacher:
    def __init__(self):
        self.surname = None
        self.name = None
        self.patronymic = None
        self.age = None
        self.education = None
        self.experience = None
        self.discipline = None


class StudyGroup:
    def __init__(self):
        self.number = None
        self.progress = None
        self.specialty = None
        self.mark = None


class College:
    def __init__(self):
        self.abbreviation = None
        self.discipline = None
        self.license = None

class Exam:
    def __init__(self):
        self.subject = None
        self.mark = None
        self.teacher = None


class StudentOnExam:
    def __init__(self):
        self.student = None
        self.mark = None
        self.teacher = None


class Car:
    def __init__(self):
        self.engine = None
        self.color = 'white'
        self.brand = None
        self.mileage = None


user_1 = Student()
user_1.surname = 'Рычкова'
print(user_1, 'surname:', user_1.surname, 'age:', user_1.age, 'birthday:', user_1.birthday)

user_1.age = 20
user_1.birthday = '20.20.2000'
print(user_1, 'surname:', user_1.surname, 'age:', user_1.age, 'birthday:', user_1.birthday)

user_2 = Car()
user_2.brand = 'Toyota'
user_2.mileage = '42141'
print(user_2, user_2.brand, user_2.mileage, user_2.color)

user_2.engine = 2.0
print(user_2, user_2.brand, user_2.mileage, user_2.engine)
