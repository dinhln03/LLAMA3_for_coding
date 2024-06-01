from CtCI_Custom_Classes.stack import Stack


class SetOfStacks:
    def __init__(self, capacity):
        self.capacity = capacity
        self.stacks = []

    def get_last_stack(self):
        if not self.stacks:
            return None
        return self.stacks[-1]

    def is_empty(self):
        last = self.get_last_stack()
        return not last or last.is_empty()

    def pop(self):
        last = self.get_last_stack()
        if not last:
            return None
        v = last.pop()
        if last.get_size() == 0:
            del self.stacks[-1]
        return v

    def push(self, data):
        last = self.get_last_stack()
        if last and not last.is_full():
            last.push(data)
        else:
            stack = Stack(self.capacity)
            stack.push(data)
            self.stacks.append(stack)
