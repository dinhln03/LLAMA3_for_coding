"""
Demonstrate differences between __str__() and __reper__().
"""

class neither:
    pass

class stronly:
    def __str__(self):
        return "STR"
    
class repronly:
    def __repr__(self):
        return "REPR"
    
class both(stronly, repronly):
    pass

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Person({0.name!r}, {0.age!r})".format(self)