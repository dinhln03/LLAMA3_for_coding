#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
函数
在python中函数默认的返回对象是None
"""


# 默认返回值为None
def hello():
    print("Hello World!")


print(type(hello()))


# 可以返回多个对象，默认是元组
def foo():
    return ['xyz', 1000, -98.6]


x, y, z = foo()
print(x, y, z)


# 关键字参数
def foo1(x):
    print(x)


foo1(x='abc')

"""
创建函数
def function_name(arguments):
    "function documentation string"
    function body suite

"""


def helloSomeOne(who):
    """hello to someone"""
    print("hello" + who)


print(helloSomeOne.__doc__)

"""
 内部/内嵌函数
如果内部函数的定义包含了在外部函数里定义的对象的引用，内部函数被称为闭包
"""


def fo():
    def ba():
        print("ba called")

    print("fo called")
    ba()


fo()

"""
传递函数
函数是可以被引用的(访问或者以其他变量作为别名)
对对象是函数，这个对象的所有别名都是可以调用的
"""


def foo():
    print("in foo()")


bar = foo
bar()


def convert(func, seq):
    return [func(eachNum) for eachNum in seq]


myseq = (123, 45.67, -6.2e8, 999999L)
print(convert(int, myseq))
print(convert(float, myseq))
