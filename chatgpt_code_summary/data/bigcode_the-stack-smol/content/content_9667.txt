import mido
import json
import time
from math import floor

import board
import busio
import digitalio
import adafruit_tlc5947


def playMidi(song_name):
    mid = mido.MidiFile('midifiles/' + song_name)

    notesDict = {'songName': 'testname', 'bpm': 999, 'notes': []}
    tempo = 0
    length = 0
    notesArray = [[]]
    tickLength = 0

    SCK = board.SCK
    MOSI = board.MOSI
    LATCH = digitalio.DigitalInOut(board.D5)

    # Initialize SPI bus.
    spi = busio.SPI(clock=SCK, MOSI=MOSI)

    # Initialize TLC5947
    tlc5947 = adafruit_tlc5947.TLC5947(spi, LATCH, auto_write=False,
                                       num_drivers=4)
    for x in range(88):
        tlc5947[x] = 0
    tlc5947.write()

    for msg in mid:
        if msg.is_meta and msg.type == 'set_tempo':
            tempo = int(msg.tempo)
            length = int(floor(mido.second2tick(mid.length,
                                                mid.ticks_per_beat,
                                                tempo)))
            tickLength = mido.tick2second(1, mid.ticks_per_beat, tempo)
            break

    print('Tick length: ' + str(tickLength))
    currentTick = 0
    notesArray[0] = [0 for x in range(89)]
    lineIncrement = 0
    for msg in mid:
        #print(msg)
        if msg.type is 'note_on' or msg.type is 'note_off':
            delayAfter = int(floor(mido.second2tick(msg.time, mid.ticks_per_beat, tempo)))
            if delayAfter == 0:
                if msg.note < 89:
                    notesArray[lineIncrement][msg.note - 12] = msg.velocity
            else:
                notesArray[lineIncrement][88] = delayAfter
                notesArray.append([0 for x in range(89)])
                lineIncrement += 1
                
                
                
            """ Old code:
                for x in range (newNote['delayAfter']):
                    if x != 0:
                        notesArray[x+currentTick] = notesArray[x+currentTick-1]
                currentTick += newNote['delayAfter']
                
            notesArray[currentTick][newNote['note'] - 1] = newNote['velocity']
            # tlc5947.write()
            notesDict['notes'].append(newNote)
            """
            
    """
    with open('notes.json', 'w') as outfile:
        json.dump(notesDict, outfile)
    """
    
    startTime = time.time()
    tlc5947.write()
    time.sleep(3)
    for line in notesArray:
        
        """
        tlc5947[27] = 900
        tlc5947[68] = 4000
        tlc5947.write()
        time.sleep(2)
        tlc5947[27] = 0
        tlc5947[68] = 0
        tlc5947.write()
        time.sleep(2)
        """
        
        print(line)
        # send array to PWM IC
        for x in range(len(line) - 1):
            if line[x] != 0:
                tlc5947[x] = line[x] * 32
            else:
                tlc5947[x] = 0
        tlc5947.write()
        # time.sleep(tickLength)
        
        time.sleep(mido.tick2second(line[88], mid.ticks_per_beat, tempo) * 0.4)
        
        for x in range(88):
            tlc5947[x] = 0
        tlc5947.write()
        
        time.sleep(mido.tick2second(line[88], mid.ticks_per_beat, tempo) * 0.6)
        
    for x in range(88):
        tlc5947[x] = 0
    tlc5947.write()


#playMidi('twinkle_twinkle.mid')
#playMidi('for_elise_by_beethoven.mid')
# playMidi('debussy_clair_de_lune.mid')
# playMidi('chopin_minute.mid')
# playMidi('jules_mad_world.mid')
