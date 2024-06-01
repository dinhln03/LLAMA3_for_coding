"""
Min Stack
-----
A LIFO abstract data type that serves as a collection of elements.
Supports retrieving the min from the stack in constant time.

"""

class MinStack(object):

	def __init__(self):
		"""
		Attributes:
			data (arr): data stored in the stack
			minimum (arr): minimum values of data stored
		"""
		self.data = []
		self.minimum = []

	def empty(self):
		"""
		Returns whether or not the stack is empty.

		Time Complexity: O(1)
		
		Returns:
			bool: whether or not the stack is empty
		"""
		return len(self.data) == 0

	def push(self, x):
		"""
		Pushes an element onto the stack.

		Time Complexity: O(1)

		Args:
			x: item to be added
		"""
		self.data.append(x)
		if not self.minimum or x <= self.minimum[-1]:
			self.minimum.append(x)

	def pop(self):
		"""
		Pops an element off the stack. 

		Time Complexity: O(1)

		Returns:
			any: the last element on the stack

		"""
		x = self.data.pop()
		if x == self.minimum[-1]:
			self.minimum.pop()
		return x

	def peek(self):
		"""
		Returns the last item on the stack but doesn't remove it.

		Time Complexity: O(1)

		"""
		return self.data[-1]

	def peek_min(self):
		"""
		Returns the min on the stack but doesn't remove it.

		Time Complexity: O(1)

		"""
		return self.minimum[-1]
