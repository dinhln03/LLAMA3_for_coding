
from pydub import AudioSegment
from pydub.playback import play
import os
import utils


class audiofile:
    

    def __init__(self, file):
        """ Init audio stream """ 
        self.file = file

    def play(self):
        """ Play entire file """
        utils.displayInfoMessage('Playing Audio')
        pathparts = self.file.rsplit(".", 1)
        fileformat = pathparts[1]
        song = AudioSegment.from_file(self.file, format=fileformat)
        play(song)
        utils.displayInfoMessage('')
        utils.displayErrorMessage('')

    def length(self):
        pathparts = self.file.rsplit(".", 1)
        fileformat = pathparts[1]
        song = AudioSegment.from_file(self.file, format=fileformat)
        return song.duration_seconds