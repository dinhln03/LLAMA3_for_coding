#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Config(object):
    DEBUG = True
    RELOADER = True
    PORT = 8080

class DevelopmentConfig(Config):
    pass

class ProductionConfig(Config):
    DEBUG = False
    RELOADER = False
