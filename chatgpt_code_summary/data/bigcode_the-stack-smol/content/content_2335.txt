from twisted.internet import defer
from signing.processor import expose

class SayHiImplementation(object):
    """
    Responds with 'hello, %s' % arg
    """
    @expose
    def say_hi(self, identifier):
        d = defer.Deferred()
        d.callback('hello, %s' % identifier)
        return d
