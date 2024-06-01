# ***************************************************************************************
# ***************************************************************************************
#
#		Name : 		importcode.py
#		Author :	Paul Robson (paul@robsons.org.uk)
#		Date : 		12th March 2019.
#		Purpose :	Import code into buffer area
#
# ***************************************************************************************
# ***************************************************************************************

import sys
from imagelib import *
#
#		Initialise and get info
#
image = BinaryImage()
bufferInfo = image.sourcePages()
firstSourcePage = bufferInfo[0]
sourcePageCount = bufferInfo[1]
pageSize = image.getBufferSize()
#
#		Clear all buffers
#
for p in range(firstSourcePage,firstSourcePage+sourcePageCount*2,2):
	for a in range(0xC000,0x10000,pageSize):
		image.write(p,a,0x80)
		image.write(p,a+pageSize-1,0x00)

print("Found and erased {0} buffers for import ${1:02x}-${2:02x}.".	\
		format(int(sourcePageCount*16384/pageSize),firstSourcePage,firstSourcePage+sourcePageCount*2-2))
#
#		Info on first buffer
#
currentPageNumber = firstSourcePage
currentPageAddress = 0xC000
currentBasePageAddress = 0xC000
bytesRemaining = pageSize
count = 1
#
#		Work through all the source
#
for f in sys.argv[1:]:
	src = [x if x.find("//") < 0 else x[:x.find("//")] for x in open(f).readlines()]
	src = " ".join([x.replace("\t"," ").replace("\n"," ") for x in src])
	src = [x for x in src.split(" ") if x != ""]
	for word in src:
		#
		#	For each word, look at it to see if it has a tag. Default is compilation.
		#
		tag = 0x40 										# Green (compile) 		$40
		if word[0] == ":":								# Red (define) 			$00
			tag = 0x00
			word = word[1:]
		elif word[0] == "[" and word[-1] == "]": 		# Yellow (execute)		$80
			tag = 0x80
			word = word[1:-1]
		#
		#	Make the final word and check it fits.
		#
		assert len(word) < 32,"Word too long "+word
		if len(word) + 4 >= bytesRemaining:				# it doesn't fit.
			image.write(currentPageNumber,currentPageAddress,0x80)
			currentPageAddress = (currentBasePageAddress + pageSize) & 0xFFFF
			if currentPageAddress == 0:
				currentPageNumber += 1
				currentPageAddress = 0xC000
			currentBasePageAddress = currentPageAddress
			count += 1
			bytesRemaining = pageSize
		#
		#print("\t\t{0:02x} {1:16} ${2:02x}:${3:04x} {4}".format(tag,word,currentPageNumber,currentPageAddress,bytesRemaining))
		#
		#	Store the word
		#
		image.write(currentPageNumber,currentPageAddress,tag+len(word))
		currentPageAddress += 1
		for c in word:
			image.write(currentPageNumber,currentPageAddress,ord(c))
			currentPageAddress += 1
		bytesRemaining = bytesRemaining - 1 - len(word)
		#
		#	Add a trailing $80 in case it is the last.
		#
		image.write(currentPageNumber,currentPageAddress,0x80)
	print("\tImported file '{0}'.".format(f))
#		
#		and write out
#
image.save()
print("Filled {0} buffers.".format(count))
