# -*- coding: utf-8 -*-
import os


print '-------操作文件和目录-------'

# 操作系统名字
print os.name + '\n'

print '\n' + '详细的系统信息'
print os.uname()

print '\n' + '环境变量'
print os.environ

print '\n' + '获取某个环境变量的值'
print os.getenv('PATH')

print '\n'


# 查看当前目录的绝对路径:
print os.path.abspath('.')
selfAbsPath = os.path.abspath('.')

# 在某个目录下创建一个新目录，
# 首先把新目录的完整路径表示出来:
filePathDir = os.path.join(selfAbsPath, 'testdir')
# '/Users/michael/testdir'

# # 然后创建一个目录:
os.mkdir(filePathDir)

# # 删掉一个目录:
os.rmdir(filePathDir)

print '-------os.path.join()函数-------'
# 这样可以正确处理不同操作系统的路径分隔符

print '-------os.path.split() 直接让你得到文件扩展名-------'
print os.path.split('/Users/michael/testdir/file.txt')

# 对文件重命名:
# os.rename('test.txt', 'test.py')

# 删掉文件:
# os.remove('test.py')

print '-------shutil-------'
# shutil模块提供了copyfile()的函数，你还可以在shutil模块中找到很多实用函数，它们可以看做是os模块的补充。

# 当前目录下的所有目录
print[x for x in os.listdir('.') if os.path.isdir(x)]

# # 当前文件夹下所有python文件
# print [x for x in os.listdir('.') if os.path.isfile(x) and
# os.path.splitext(x)[1]=='.py']

# print os.listdir('.')

# print dir(os.path)

# 编写一个search(s)的函数，能在当前目录以及当前目录的所有子目录下查找文件名包含指定字符串的文件，并打印出完整路径：


def search(fileName):
    currentPath = os.path.abspath('.')
    for x in os.listdir('.'):
        if os.path.isfile(x) and fileName in os.path.splitext(x)[0]:
            print x
        if os.path.isdir(x):
            newP = os.path.join(currentPath, x)
            print newP


print '-------search start-------'
search('0810')
