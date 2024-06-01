"""
多线程操作共享的全局变量是不安全的，多线程操作局部 只归某个线程私有，其他线程是不能访问的
"""
import threading


def do_sth(arg1, arg2, arg3):
    local_var1 = arg1
    local_var2 = arg2
    local_var3 = arg3

    fun1(local_var1, local_var2, local_var3)
    fun2(local_var1, local_var2, local_var3)
    fun3(local_var1, local_var2, local_var3)


def fun1(local_var1, local_var2, local_var3):
    print('%s: %s -- %s -- %s' % (threading.current_thread().name, local_var1,
                                  local_var2, local_var3))


def fun2(local_var1, local_var2, local_var3):
    print('%s: %s -- %s -- %s' % (threading.current_thread().name, local_var1,
                                  local_var2, local_var3))


def fun3(local_var1, local_var2, local_var3):
    print('%s: %s -- %s -- %s' % (threading.current_thread().name, local_var1,
                                  local_var2, local_var3))


t1 = threading.Thread(target=do_sth, args=('a', 'b', 'c'))
t2 = threading.Thread(target=do_sth, args=('d', 'e', 'f'))

t1.start()
t2.start()
