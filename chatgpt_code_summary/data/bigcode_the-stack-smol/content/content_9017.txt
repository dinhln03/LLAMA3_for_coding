#!/usr/bin/env python
# coding: utf-8

# list

classMates = ['Micheal', 'Lucy', 'Anna']
print classMates

# 获取长度
print len(classMates)

# 取值
print classMates[2]

print classMates[-1]

# 追加
classMates.append('Adam')
print classMates

# 插入
classMates.insert(1, 'Paul')
print classMates

# 删除
classMates.pop()
print classMates
classMates.pop(0)
print classMates

# 替换
classMates[1] = 'Sam'
print classMates

# 类型无需一致
L = ['Anna', 22, True]
print L

# 嵌套
s = ['Python', 'Ruby', ['Java', 'objc']]
print s
print s[2][1]


# tuple

classMates = ('Micheal', 'Lucy', 'Adam')
print classMates

# 因为tuple不可变，所以代码更安全。如果可能，能用tuple代替list就尽量用tuple

t = (1, 2)
print t

t = ()
print t

t = (1)
print t

t = (1,)
print t

t = ('a', 'b', ['A', 'B'])
t [2][0] = 'X'
t [2][1] = 'Y'
print t
