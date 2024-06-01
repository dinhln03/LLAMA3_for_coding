#!/usr/bin/env python


'''
Created on Apr 12, 2017

@author: Brian Jimenez-Garcia
@contact: brian.jimenez@bsc.es
'''

import sys
import os
if len(sys.argv[1:]) != 2:
    raise SystemExit("usage: %s pdb_file1 pdb_file2" % os.path.basename(sys.argv[0]))

pdb_file1 = sys.argv[1]
pdb_file2 = sys.argv[2]

# Panda3D imports
from pandac.PandaModules import loadPrcFileData
from emol import EMol

width = 1400
height = 900

# Change window properties
loadPrcFileData("", "window-title Energy Visualizer")
loadPrcFileData("", "fullscreen 0")
loadPrcFileData("", "win-size %s %s" % (width, height))

from direct.showbase.ShowBase import ShowBase
base = ShowBase()

# Set up a loading screen
from direct.gui.OnscreenText import OnscreenText,TextNode
loadingText=OnscreenText("Loading molecules...",1,fg=(1,1,1,1),
                         pos=(0,0),align=TextNode.ACenter,
                         scale=.07,mayChange=1)
# Render three frames to avoid black screen
base.graphicsEngine.renderFrame() 
base.graphicsEngine.renderFrame()
base.graphicsEngine.renderFrame()

# Load the game
visualizer = EMol(width, height, pdb_file1, pdb_file2)

# Hide loading
loadingText.cleanup() 

base.run()
