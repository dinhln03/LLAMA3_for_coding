import QueueLinkedList as queue

"""
    n1
    /\
 n2    n3
 /\    /\
n4 n5 n6 n7
"""


class BinaryTree:
    def __init__(self, size) -> None:
        self.customList = size * [None]
        self.lastUsedIndex = 0
        self.maxSize = size

    def inserNode(self, value):
        if self.lastUsedIndex + 1 == self.maxSize:
            return "Full"
        self.customList[self.lastUsedIndex + 1] = value
        self.lastUsedIndex += 1
        return "Inserted"

    def searchNode(self, value):
        if value in self.customList:
            return "Success"
        return "Not found"

    def preOrderTraversal(self, index):
        # root -> left -> right
        if index > self.lastUsedIndex:
            return
        print(self.customList[index])
        self.preOrderTraversal(index * 2)
        self.preOrderTraversal(index * 2 + 1)

    def inOrderTraversal(self, index):
        # left -> root -> right
        if index > self.lastUsedIndex:
            return
        self.inOrderTraversal(index * 2)
        print(self.customList[index])
        self.inOrderTraversal(index * 2 + 1)

    def postOrderTraversal(self, index):
        # left -> right -> root
        if index > self.lastUsedIndex:
            return
        self.postOrderTraversal(index * 2)
        self.postOrderTraversal(index * 2 + 1)
        print(self.customList[index])

    def levelOrderTraversal(self, index):
        for i in range(index, self.lastUsedIndex + 1):
            print(self.customList[i])

    def deleteNode(self, value):
        if self.lastUsedIndex == 0:
            return "Nothing to delete"
        for i in range(1, self.lastUsedIndex + 1):
            if self.customList[i] == value:
                self.customList[i] = self.customList[self.lastUsedIndex]
                self.customList[self.lastUsedIndex] = None
                self.lastUsedIndex -= 1
                return "Deleted"

    def deleteTree(self):
        self.customList = None
        return "Deleted"


newBT = BinaryTree(8)
print(newBT.inserNode("N1"))
print(newBT.inserNode("N2"))
print(newBT.inserNode("N3"))
print(newBT.inserNode("N4"))
print(newBT.inserNode("N5"))
print(newBT.inserNode("N6"))
print(newBT.inserNode("N7"))
print(newBT.inserNode("N8"))

print(newBT.searchNode("N1"))
print(newBT.searchNode("N8"))

print("preOrderTraversal")
newBT.preOrderTraversal(1)

print("inOrderTraversal")
newBT.inOrderTraversal(1)

print("postOrderTraversal")
newBT.postOrderTraversal(1)

print("levelOrderTraversal")
newBT.levelOrderTraversal(1)

print(newBT.deleteNode("N4"))
newBT.levelOrderTraversal(1)

print(newBT.deleteTree())
