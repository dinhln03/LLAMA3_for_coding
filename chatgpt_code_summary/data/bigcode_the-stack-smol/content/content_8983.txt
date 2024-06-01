# -*- coding: utf-8 -*-

class C:
	a = 'abc'
	def __getattribute__(self, args):
		print('__getattribute_ is called')
		#import pdb; pdb.set_trace()
		#return object.__getattribute__(self, args)
		return super().__getattribute__(args)
	def __getattr__(self, name):
		print('__getattr()__ is called')
		return name+ 'from __getattr__'
	def __get__(self, instance, owner):
		print('__get__() is called', instance,'||', owner, '||')
		return self
	def __set__(self, instance, value):
		print('__set__() is called', instance, value)
	def foo(self, x):
		print('foo:',x)
	def __call__(sef, *args, **kwargs):
		print('__call()__ is called', args, kwargs)

class C2:
	d = C()
