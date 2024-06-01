__author__ = 'Gobin'

from redditcli.api import base


class Account(base.Resource):
    resource_name = 'Account'


class AccountManager(base.ResourceManager):
    resource_class = Account

    def me(self):
        return self._get('/api/v1/me')

    def getkarma(self):
        return self._get('/api/v1/me/karma')

    def getfriends(self):
        return self._get('/api/v1/me/friends', 'data')

    def getprefs(self):
        return self._get('/api/v1/me/prefs')

    def gettrophies(self):
        return self._get('/api/v1/me/trophies')