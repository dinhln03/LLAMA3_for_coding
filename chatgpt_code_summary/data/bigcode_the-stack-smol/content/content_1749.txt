# 右侧加法和原处加法: __radd__和__iadd__
"""
__add__并不支持+运算符右侧使用实例对象。要实现一并编写__radd__方法。
只有当+右侧的对象是实例，而左边对象不是类实例时，Python才会调用__radd++，
在其他情况下则是由左侧对象调用__add__方法。

"""


class Commuter:
    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        # 如果没有instance测试，当两个实例相加并且__add__触发
        # __radd__的时候，我们最终得到一个Commuter,其val是另一个Commuter
        if isinstance(other, Commuter): other = other.val
        print("add")
        return self.val + other

    def __radd__(self, other):
        print("radd")
        # 注意和__add__顺序不一样
        return other + self.val


# 原处加法 编写__iadd__或__add__如果前者空缺使用后者
class Number:
    def __init__(self, val):
        self.val = val

    def __add__(self, other):
        return Number(self.val + other)


x = Commuter(89)
y = Commuter(99)
print(x + 1)
print(x + y)

X = Number(5)
X += 1
X += 1
print(X.val)