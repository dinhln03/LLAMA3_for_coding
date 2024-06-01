"""
A validator for a frontend failure model. The model contains all
the failing web frontends and their status, as well as the virtual
machines they run on.
"""
from vuluptuous import Schema

schema = Schema({
'web_frontends_failures'
})
