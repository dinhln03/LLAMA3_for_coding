#!/usr/bin/env python

import numpy as np
import sys
sys.path.append("../ar/")
import fastopc, time
import functionLib as lib
import micStream
nStrips = 16
lStrip  = 64
client = fastopc.FastOPC('localhost:7890')

pixels    = lib.Pixels(nStrips, lStrip, 0)
theoStrip = np.zeros([lStrip, 3])

stream = micStream.Stream(fps=40, nBuffers=4)

powerSmooth = lib.ExpFilter(val=0.05, alpha_rise=0.05, alpha_decay=0.05)
nColorWheel = 1500
colorWheel = lib.getColorWheel(nColorWheel)
frameCount = 0

while True:
    success = stream.readAndCalc()
    if success:
        frameNumEff = np.mod(frameCount, nColorWheel)
        power = np.sum(stream.freqSpectrum[10//5:300//5])
        powerSmooth.update(power)
        displayPower = int(122*power/powerSmooth.value)
        theoStrip = displayPower * colorWheel[frameNumEff]
        pixels.update(theoStrip, 0.9, 0.2)
        #print(displayPower * colorWheel[frameNumEff])
        client.putPixels(0, pixels.getArrayForDisplay())
        frameCount+=1
