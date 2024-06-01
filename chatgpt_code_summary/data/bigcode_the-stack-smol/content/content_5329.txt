import zutils

class zhighlighter:
	def highlight(self, text):
		return [(zutils.CL_FG, zutils.CL_BG, zutils.AT_BLINK if i % 2 == 0 else zutils.AT_NORMAL) for i in range(len(text))] #LOL!
	