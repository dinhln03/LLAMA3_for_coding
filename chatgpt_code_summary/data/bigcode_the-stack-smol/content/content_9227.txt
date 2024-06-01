import time
import asyncore

from ardos.instance.InstanceObject import InstanceObject
from ardos.core.ArdosServer import ArdosServer

class TestObjectIO(InstanceObject):

	def __init__(self, parent):
		InstanceObject.__init__(self, parent, 'DistributedTestObject')

	def getTestMethod(self):
		return [None, None]


class TestServerIO(ArdosServer):

	def __init__(self):
		ArdosServer.__init__(self)

		self.dcManager.loadDCFile('test.dc.json')

		self.connect(('127.0.0.1', 7199))

		self.testObject = TestObjectIO(self)

		asyncore.loop()

	def handleConnect(self):
		ArdosServer.handleConnect(self)

		self.generateInstanceObject(self.testObject, 1, 1)


if __name__ == '__main__':
	test = TestServerIO()