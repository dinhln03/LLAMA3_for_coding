import timeit

mapx = 512
mapy = 512
# Good seeds:
#	772855	Spaced out continents
#	15213	Tight continents
#	1238	What I've been working with, for the most part
#	374539	Sparse continents
#	99999
seed = 773202
sea_level = 0.6

DEBUG = 0
GFXDEBUG = 0

setup_time = timeit.default_timer()

tiles = [[None] * mapx for _ in range(mapy)]
lands = []
towns = []
countries = []
have_savefile = False

class Clock():
	def __init__(self,t):
		self.time_minutes = t  

	def inc(self,t):
		self.time_minutes += t
		self.time_minutes = self.time_minutes % (60*24)

	def fmt_time(self):
		m = self.time_minutes % 60
		h = self.time_minutes // 60
		return ("%02d%02dZ" % (h, m))

clock = Clock(9*60) # 9 AM