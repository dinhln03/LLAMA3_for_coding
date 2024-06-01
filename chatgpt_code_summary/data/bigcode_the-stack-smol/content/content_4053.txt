class Employee:
    def __init__(self, fname, lname):
        self.fname = fname
        self.lname = lname
        # self.email = f"{fname}.{lname}@sandy.com"

    def explain(self):
        return f"This employee is {self.fname} {self.lname}"
    def email(self):
        return f"{self.fname}.{self.lname} @parker.com"

obj1 = Employee("Peter", "Parkar")

print(obj1.email())

obj1.fname = "Spider"
print(obj1.email())         #required call email() function to print

