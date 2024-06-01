#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import glob
import math
from EMAN2 import *

def file_base(movie):
	# return the filename and basename, exclude '.p3'
	return movie, os.path.basename(os.path.splitext(movie)[0]).replace('.p3', '')

def check(log,c_p):
	with open(log) as log_r:
		lines = [line for line in log_r]
	x0 = 0
	y0 = 0
	f = c_p['throw']
	bad = []
	while len(lines) > 0:
		line1 = lines.pop(0)
		if "...... Frame (" in line1:
			line = line1.strip().split()
			x = float(line[-2])
			y = float(line[-1])
			if math.sqrt((x - x0)**2 + (y - y0)**2) * c_p['apixr'] > c_p['target']:
				bad += [f]
			f += 1					
			x0 = x
			y0 = y
	return bad

def run_motioncor2(movie, c_p):
	movie, basename = file_base(movie)
	# generate the com file
	out = basename+'_throw{:03}'.format(c_p['throw'])
	o_com = out + '.com'
	o_log = out + '.log'
	o_mrc = out + '.mrc'
	common = 'motioncor2 -InMrc {} -OutMrc {} -Iter 10 -Bft 100 -FtBin {} -Throw {} -FmRef -1 -Tilt {} {}'.format(movie,o_mrc,c_p['bin'],c_p['throw'],c_p['tilt'], c_p['gainref'])
	with open(o_com, 'w') as o_com_w:
		if c_p['local'] == 0:
			o_com_w.write('{} -Patch 0 0'.format(common))
		else:
			o_com_w.write('{} -Patch {} {} -LogFile {} -FmDose {} -PixSize {} -kV {}'.format(common,c_p['patch'],c_p['patch'],out+'_',c_p['dose'],c_p['apixr'],c_p['voltage']))
	# run the com
	with open(o_log, 'w') as write_log:
		subprocess.call(['sh', o_com], stdout=write_log, stderr=subprocess.STDOUT)
	# check the shifts
	bad = check(o_log,c_p)
	# decide bad
	decide(movie, bad, c_p)

def decide(movie, bad, c_p):
	if bad == []:
		if c_p['local'] == 0:
			print "No bad frames. Do local now."
			c_p['local'] = 1
			run_motioncor2(movie, c_p)
		else:
			print "No bad frames. Local done for {}. Throwed the first {} frames.".format(movie, c_p['throw'])
	elif max(bad) < c_p['maxthrow']:
		c_p['throw'] = max(bad)
		print "Throw the first {} frames.".format(c_p['throw']), "Bad frames: ", bad
		run_motioncor2(movie, c_p)		
	else: # if too many bad frames
		print '{} has too many bad frames: '.format(movie), bad

def main():
	progname = os.path.basename(sys.argv[0])
	usage = progname + """ [options] <movies>
	Output unfiltered and filtered sum using MotionCor2.
	Automatically discard bad frames.
	Needs:
	'motioncor2' command (v1, Zheng et al., 2017)
	'EMAN2' python module (v2.11, Tang et al., 2007)
	"""
	
	args_def = {'apix':1.315, 'apixr':0.6575, 'bin':1, 'patch':5, 'voltage':300, 'time':200, 'rate':7, 'target':5, 'tilt':'0 0', 'gainref':''}
	parser = argparse.ArgumentParser()
	parser.add_argument("movie", nargs='*', help="specify movies (mrc, mrcs, dm4) to be processed")
	parser.add_argument("-a", "--apix", type=float, help="specify counting apix, by default {}".format(args_def['apix']))
	parser.add_argument("-ar", "--apixr", type=float, help="specify real apix of input movie, by default {}".format(args_def['apixr']))
	parser.add_argument("-b", "--bin", type=float, help="specify binning factor, by default {}".format(args_def['bin']))
	parser.add_argument("-p", "--patch", type=int, help="specify the patch, by default {}".format(args_def['patch']))
	parser.add_argument("-v", "--voltage", type=int, help="specify the voltage (kV), by default {}".format(args_def['voltage']))
	parser.add_argument("-t", "--time", type=float, help="specify exposure time per frame in ms, by default {}".format(args_def['time']))
	parser.add_argument("-r", "--rate", type=float, help="specify dose rate in e/pix/s (counting pixel, not superresolution), by default {}".format(args_def['rate']))
	parser.add_argument("-ta", "--target", type=float, help="specify the target resolution, by default {}".format(args_def['target']))
	parser.add_argument("-ti", "--tilt", type=str, help="specify the tilt, by default {}".format(args_def['tilt']))
	parser.add_argument("-g", "--gainref", type=str, help="specify the gainref option, by default {}. e.g., '-Gain ../14sep05c_raw_196/norm-amibox05-0.mrc -RotGain 0 -FlipGain 1'".format(args_def['gainref']))
	args = parser.parse_args()
	
	if len(sys.argv) == 1:
		print "usage: " + usage
		print "Please run '" + progname + " -h' for detailed options."
		sys.exit(1)
	# get default values
	for i in args_def:
		if args.__dict__[i] == None:
			args.__dict__[i] = args_def[i]
	# get common parameters
	dose = args.time/1000.0 * args.rate / args.apix ** 2
	voltage = args.voltage
	c_p = {'dose':dose, 'apix':args.apix, 'apixr':args.apixr, 'bin':args.bin, 'patch':args.patch, 'voltage':voltage, 'target':args.target, 'tilt':args.tilt, 'throw':0, 'gainref':args.gainref}
	# loop over all the input movies
	for movie in args.movie:
		if movie[-3:] == '.gz':
			subprocess.call(['gunzip', movie])
			movie = movie[:-3]
		basename = os.path.basename(os.path.splitext(movie)[0])
		suffix = os.path.basename(os.path.splitext(movie)[1])
		basename_raw = basename
		# unify mrc and mrcs to mrcs format
		m = basename+'.p3.mrcs'
		if suffix in ['.mrc','.mrcs']:
			os.symlink(movie, m)
			movie, basename = file_base(m)
		# get nimg
		c_p['nimg'] = EMUtil.get_image_count(movie)
		# convert dm4 to mrcs
		if suffix == '.dm4':
			for i in xrange(c_p['nimg']):
				d=EMData(movie, i)
				d.write_image(m, i)
			movie, basename = file_base(m)
		# here we assume 36e is the maximal dose that still contributes to visualization of protein side chains, and a total of 20e is the minimum to ensure good alignment. therefore, you can throw the first 16e at most.
		c_p['maxthrow'] = min(16/dose, c_p['nimg'] - 20/dose)
		# motioncor2
		c_p['local'] = 0 #0 means no local, only global
		c_p['throw'] = 0
		run_motioncor2(movie, c_p)
		# delete intermediate files, they contain '.p3.'
		for i in glob.glob(basename_raw + '*.p3.*'):
			os.unlink(i)
			
if __name__ == '__main__':
	main()
