# -*- coding: utf-8 -*-
'''   Globals variables  '''

# ##################################################################################
# MG ILLUMINATION                                                           	   #
# First Crazy Debroussailleur : jDepoortere                                        #
# Author : cPOTTIER                                                                #
# Date : 12-05-2016                                                                #
# ##################################################################################

# Required modules
import sys
import os 

# InK modules
import graphs
import nomen
import ink.proto
import ink.query
import ink.io
import nask.sdk
import nask.sdk.casting
import nask.sdk.shots as shots
import nask.sdk.hit	
import proj.pipe.ink.graphs as prodgraphs
from subprocess import Popen, PIPE

# Maya modules
import sip

# Optionals modules
import re
import shutil
import time
import datetime
from datetime import datetime
import subprocess
import glob
import json
import shutil
import string
import subprocess
import collections
from collections import OrderedDict

# QT modules
from PyQt4 import QtGui
from PyQt4 import QtGui, QtCore, Qt, QtOpenGL
from PyQt4.QtCore import QThread

# qt module for InK
try:
	if 'sandboxQt' in sys.modules:
		del(sys.modules["sandboxQt"])
		import sandboxQt 
	else:
		import sandboxQt
except:
	pass


# Globals
CONNECT_USER_INFOS      = ink.io.ConnectUserInfo()
CONNECT_USER0           = CONNECT_USER_INFOS[0]
CONNECT_USER1           = CONNECT_USER_INFOS[1] # todo to ask why ?
PROJECT                 = CONNECT_USER_INFOS[2].upper() # cf Nomen.GetFilm()
projectLower            = PROJECT.lower()
USER                    = CONNECT_USER_INFOS[1]
MAIL_HOSTNAME           = 'HOSTNAME.illum-mg.fr'
MAIL_USER               = USER+'@illum-mg.fr'
LOCALPATH               = '/u/'+projectLower+'/Users/'+USER+'/Presets/Graphs/'


# Useful Classes
if str(USER) == 'cpottier': # for dev
	if '__InK__classes_forDev' in sys.modules:
		del(sys.modules["__InK__classes_forDev"])
		import __InK__classes_forDev
		from __InK__classes_forDev import __PIPEIN_GRAPH__
	else:
		import __InK__classes_forDev
		from __InK__classes_forDev import __PIPEIN_GRAPH__
else:
	if '__InK__classes' in sys.modules:
		del(sys.modules["__InK__classes"])
		import __InK__classes
		from __InK__classes import __PIPEIN_GRAPH__
	else:
		import __InK__classes
		from __InK__classes import __PIPEIN_GRAPH__
	print sys.modules
