# coding:utf8

from flask import request


routes = dict()


class ApiServiceBase(type):
	def __new__(cls, name, base, attrs):
		# super(type, obj) require isinstance(obj, type)
		return super(ApiServiceBase, cls).__new__(cls, name, base, attrs)

	def __init__(self, name, base, attrs):
		if name == 'ApiService':
			pass
		else:
			route = '/' + self.app + '/' + self.resource
			if self.resource:
				route += '/'
			routes[route] = {
				'cls': self
			}


class ApiService(object):
	__metaclass__ = ApiServiceBase

	def handle(self):
		self.request = request
		req_method = getattr(self, self.request.method.lower(), None)
		return req_method()

