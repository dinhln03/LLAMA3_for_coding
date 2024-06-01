from stack import Stack as s

class Queue:
    def __init__(self, iter=[]):
        self.stack_one = s()
        self.stack_two = s()
        self._len = 0 

        for item in iter:
            self.enqueue(item)

    def enqueue(self, value):
        if value:
            self.stack_one.push(value)
            self._len += 1
            return self.stack_one

        return False

    def dequeue(self):
        if self._len == 0:
            return False
        else:    
            for _ in range(self._len - 2):
                self.stack_two.push(self.stack_two.pop())
                last = self.stack_one.pop()
            for _ in range(self._len - 2):
                self.stack_one.push(self.stack_two.pop())
            self._len -= 1
            return last