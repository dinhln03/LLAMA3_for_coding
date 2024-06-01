# Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

# For example, this binary tree is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
But the following is not:
    1
   / \
  2   2
   \   \
   3    3


class Solution(object):
    def help(self, el, r):
        if el == None and r == None:
            return True
        if el and r and el.val == r.val:
            return self.help(el.right, r.left) and self.help(el.left, r.right)
        return False
    
    def isSymmetric(self, root):
        if root:
            return self.help(root.left, root.right)
        return True