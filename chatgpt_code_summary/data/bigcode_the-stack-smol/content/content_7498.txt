# Most of this code is stolen from myself, but it's an app about pirating so whatev

import os
import sys
import time
import datetime
import subprocess

import pycurl
import spotipy
import spotipy.util

from StringIO import StringIO
from collections import deque

import sys

# If this seems like overkill, that's because it is. I wrote this get_args code
# for general use, this function is not specific to this
def get_args():
    def is_flag(check):
        if check[:2] == '--':
            return True
        elif check[0] == '-':
            return True
        else:
            return False

    argdict = {}
    for i, arg in enumerate(sys.argv):
        if i == 0:
            # this is just the name of the program
            continue

        check = sys.argv[i]
        if not is_flag(check):
            continue
        elif check[:2] == '--':
            # single multi-character flag
            check = check[2:]

            # it makes my life easier if they do this
            if '=' in check:
                check = check.split('=')
                argdict[check[0]] = check[1]
                continue
            j = i+1
            params = []
            while not is_flag(sys.argv[j]):
                params.append(sys.argv[j])
                j += 1
                if j >= len(sys.argv):
                    break

            if not params:
                # if params is empty, just make the value True
                params = True

            argdict[check] = params
        elif check[0] == '-':
            # multiple single character flags
            check = check[1:]

            # it makes my life easier if they do this
            if '=' in check:
                check = check.split('=')
                argdict[check[0]] = check[1]
                continue

            if len(check) == 1:
                # if there is just one character, the arg can have params
                j = i+1
                params = []
                while not is_flag(sys.argv[j]):
                    params.append(sys.argv[j])
                    j += 1

                if not params:
                    # if params is empty, just make the value True
                    params = True

                argdict[check] = params
            else:
                for c in check:
                    argdict[c] = True
    return argdict

# Hacky AF but it works
class Foobar:
    def wait(self):
        return

class Song:
    def __init__(self, title, artist, pl):
        self.title = title.replace("/", "&").replace(" -", ",").replace("-", "~")
        self.artist = artist.replace("/", "&")
        self.playlist = pl
        self.id = None

class Pirate:
    ### CONSTRUCTOR ###
    def __init__(self, settings):
        self.downloadthreads = []
        self.downloadqueue = deque()

        self.alive = True

        self.verbose = int(settings["verbose"])

        self.downloadthreads = settings["download_threads"]

        self.dl_folder  = os.path.realpath(settings["dl_folder"])
        self.log_folder = os.path.realpath(settings["log_folder"])
        self.err_folder = os.path.realpath(settings["err_folder"])

        self.username = settings["username"]
        scope = "user-library-read"
        client_id = settings["client_id"]
        client_secret = settings["client_secret"]
        redirect_url = "http://localhost:8888/callback"

        # Request a token
        token = spotipy.util.prompt_for_user_token(self.username, scope, client_id, client_secret, redirect_url)
        self.sp = spotipy.Spotify(auth=token)

        args = get_args()

        if "plid" in args:
            self.plid = args["plid"]
        elif "p" in args:
            self.plid = args["p"]
        else:
            self.plid = None

        self.mode = "all playlists"

        if "mode" in args:
            self.mode = args["mode"]
        elif "m" in args:
            self.mode = args["m"]

        self.mode = self.mode.lower()
        if self.mode not in ["all playlists", "apl", "saved", "s", "everything", "e", "playlist", "pl"]:
            self.log_error("Invalid mode: %s" % self.mode, 0)
            self.terminate()

        if self.mode == "playlist" and self.plid is None:
            self.log_error("You need to supply a playlist to fetch", 0)
            self.terminate()

    ### THESE FUNCTIONS HANDLE STARTUP ###
    def init(self):
        if not os.path.exists(self.dl_folder):
            os.makedirs(self.dl_folder)

        #initiate logs
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        if not os.path.exists(self.err_folder):
            os.makedirs(self.err_folder)

        self.now = lambda: str(datetime.datetime.now())

        # Error log
        fname = self.now().replace(':', '-') + ".log"

        self.log_file = os.path.join(self.log_folder, fname)
        self.err_file = os.path.join(self.err_folder, fname)

        with open(self.log_file, 'w') as f:
            f.write("Activity log for yt-streamer:\n---------------------\n")
        with open(self.err_file, 'w') as f:
            f.write("Error log for yt-streamer:\n---------------------\n")

    def go(self):
        try:
            if self.mode in ["all playlists", "apl"]:
                self.fetchPls()
            elif self.mode in ["saved", "s"]:
                self.fetchSaved()
            elif self.mode in ["everything", "e"]:
                self.fetchAll()
            elif self.mode in ["playlist", "pl"]:
                playlists = self.sp.user_playlists(self.username)
                for playlist in playlists['items']:
                    if playlist['name'].lower() == self.plid.lower():
                        self.fetch(playlist)
                    else:
                        self.log("Not the one: %s" % playlist['name'], 20)
                        self.log("Looking for: %s" % self.plid, 20)
            else:
                self.log_error("Invalid mode: %s" % self.mode, 0)
                self.terminate()

            self.handle_downloading()
        except Exception as e:
            self.log_error(str(e), 10)
        except:
            self.log("Program terminated by user (Keyboard Interrupt)", 10)
            self.terminate()

    def addtrack(self, track, plname):
        name = track['name']
        artists = ""
        if len(track['artists']) > 0:
            for artist in track['artists']:
                artists += artist['name'] + ", "
            artists = artists[:-2]
        else:
            artists = "Unknown Artist"

        # Add to download queue as a Song object
        s = Song(name, artists, plname)
        s.id = self.ytsearch(s)
        self.downloadqueue.append(s)

        self.log("Found song: %s by %s" % (s.title, s.artist), 10)

    def fetchAll(self):
        self.fetchSaved()
        self.fetchPls()

    def fetchSaved(self):
        # Fetch saved music
        results = self.sp.current_user_saved_tracks()
        for item in results['items']:
            track = item['track']
            self.addtrack(track, "My Music")

    def fetchPls(self):
        self.log("Fetching all playlists of %s" % self.username, 10)
        # Fetch all spotify playlists of a user
        playlists = self.sp.user_playlists(self.username)
        for playlist in playlists['items']:
            if playlist['owner']['id'] == self.username:
                self.fetch(playlist)

    # pl is a spodipy playlist object
    def fetch(self, playlist):
        # Fetch all songs in a playlist
        self.log("Fetching playlist: %s" % playlist['name'])
        results = self.sp.user_playlist(self.username, playlist['id'], fields="tracks,next")
        tracks = results['tracks']

        while True:
            for item in tracks['items']:
                # For each item, get the artist, playlist, and song name
                track = item['track']
                self.addtrack(track, playlist["name"])

            if tracks['next']:
                tracks = self.sp.next(tracks)
            else:
                break

    def handle_downloading(self):
        # Fix so queue never mutates when this is happening
        while self.alive:
            if (len(self.downloadqueue)):
                to_push = self.downloadqueue.popleft() # Get the downloading song
                self.log("Starting download: %s" % to_push.title, 10)

                # Start download, wait for it to finish

                self.ytdl(to_push).wait()
                self.log("Finished download: %s" % to_push.title, 10)

                # Edit ogg header info
                # comment[0]="ARTIST=me";
                # comment[1]="TITLE=the sound of Vorbis";

            else:
                self.terminate()

    def ytdl(self, song):
        pl = song.playlist
        name = song.title + " - " + song.artist
        vid = song.id
        # Need better error handling (exceptions)
        # Fix it later (or never)
        if vid is None:
            return Foobar()

        # TODO: replace dl with self.dl_folder
        ndir =  "./%s/%s" % (self.dl_folder, pl)
        fname = "%s/%s.ogg" % (ndir, name)

        if not os.path.exists(ndir):
            os.makedirs(ndir)

        # check if file exists
        if not os.path.isfile(fname):
            # download the audio of a video based on the youtube id
            try:
                p = subprocess.Popen(["youtube-dl",\
                      "--extract-audio",\
                      "--audio-format", "vorbis",\
                      "-o", ndir + "/" + name + ".%(ext)s",\
                      "https://www.youtube.com/watch?v=" + vid],\
                      stdout=None)
                return p
            except Exception as e:
                return Foobar()
        return Foobar()

    def getbetween(self, search, left, right):
        for line in search.splitlines():
            if not left in line:
                continue
            out = line.split(left)[1].split(right)[0]

            return out

    def curl(self, url):
        c = pycurl.Curl()
        c.setopt(c.URL, url)

        buf = StringIO()
        c.setopt(c.WRITEDATA, buf)

        c.perform()
        c.close()

        return buf.getvalue()

    def ytsearch(self, song):
        # Get query from song
        query = song.title + " " + song.artist
        # Get the first result from the query
        try:
            query = query.replace(" ", "+")
            url = "https://www.youtube.com/results?search_query="
            url += query
            #url += '+-"music+video"+dirty'
            #url += '+lyrics+dirty+radio+edit'
            url += '+lyrics+dirty+'

            # Search for video, try to avoid music videos since they suck
            search = self.curl(url)
            left = "watch?v="
            right = "\""
            video_id = self.getbetween(search, left, right)
            return video_id
        except Exception as e:
            # This means no results (or no internet connection)
            self.log_error("Failed to ytdl for some reason", 10)
            return None

    def terminate(self):
        self.alive = False

    ### THESE FUNCTIONS HANDLE LOGGING ###

    def report(self, msg):
        sys.stdout.flush()
        print msg

        # Grab the lock
        with open(self.log_file, 'a') as f:
            f.write(self.now() + ':\t')
            f.write(msg + '\n')

        return msg

    def log(self, msg, log_level=0):
        # This function is too complicated to properly comment
        if log_level <= self.verbose:
            return self.report(msg)
        else:
            return msg # This is pythonic

    def log_error(self, e, log_level=0):
        # Grab the lock
        if log_level <= self.verbose:
            with open(self.err_file, 'a') as f:
                f.write(self.now() + ':\t')
                f.write(str(e) + ('\n'))
            self.report("An exception has been raised: %s" % (e,))
        return e

def read(filen):
    config = dict()
    try:
        with open(filen) as cfg:
            for line in cfg:
                line = line.split('#')[0] # Comments, yay!
                line = line.split('//')[0] # //Comments, yay!
                parts = line.split(':')
                if len(parts) == 2:
                    config[parts[0].strip()] = parts[1].strip()
                else:
                    pass # This is pythonic
            return config
    except Exception as e:
        print "Error opening settings file %s" % filen
        return None

settings = read("./dat/pirate.conf")
app = Pirate(settings)
app.init()
app.go()
