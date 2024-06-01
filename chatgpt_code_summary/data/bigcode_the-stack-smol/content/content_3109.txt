class OrderedList:
    def __init__(self, unique=False):
        self.list = []
        self.__unique = unique
    
    def add(self, value):
        i = 0
        while (i < len(self.list)) and (self.list[i] < value):
            i += 1
        if self.__unique:
            if len(self.list) == i or self.list[i] != value:
                self.list.insert(i, value)
        else:
            self.list.insert(i, value)

    def is_empty(self):
        return (len(self.list) == 0)

    def remove_min(self):
        if len(self.list) == 0:
            return None
        return self.list.pop(0)

    def remove_max(self):
        if len(self.list) == 0:
            return None
        return self.list.pop()

    def get_min(self):
        if len(self.list) == 0:
            return None
        return self.list[0]

    def get_max(self):
        if len(self.list) == 0:
            return None
        return self.list[-1]