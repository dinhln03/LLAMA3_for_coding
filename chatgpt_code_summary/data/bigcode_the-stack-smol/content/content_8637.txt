#!/usr/bin/env python

import sys, os
import itertools, operator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def reader(myfile):

	t = np.array([])

	print(myfile)

	with open(myfile) as f:
		lines = f.readlines()
		for line in lines:
			parts = line.split(" ")
			if(len(parts)>1):
				t=np.append(t,float(parts[7]))
		f.close()
		return t



def tuples_by_dispatch_width(tuples):
	ret = []
	tuples_sorted = sorted(tuples, key=operator.itemgetter(0))
	for key,group in itertools.groupby(tuples_sorted,operator.itemgetter(0)):
		ret.append((key, zip(*map(lambda x: x[1:], list(group)))))
	return ret

def printgraphs(results_tuples,filename,title):

	global_ws = [1,2,4,8,16,32,64]

	plt.clf()
	plt.cla()
	markers = ['.', 'o', 'v', '*', 'D']
	fig = plt.figure()
	plt.grid(True)
	plt.title(title)
	ax = plt.subplot(111)
	ax.set_xlabel("$Threads$")
	ax.set_ylabel("$Throughput(Mops/sec)$")

	i = 0
	c="b"
	tuples_by_dw = tuples_by_dispatch_width(results_tuples)
	for tuple in tuples_by_dw:
		dw = tuple[0]
		ws_axis = tuple[1][0]
		ipc_axis = tuple[1][1]
		x_ticks = np.arange(0, len(global_ws))
		x_labels = map(str, global_ws)
		ax.xaxis.set_ticks(x_ticks)
		ax.xaxis.set_ticklabels(x_labels)
		#ax.yaxis.set_ticks(np.arange(0,210,10))

		print x_ticks
		print ipc_axis
		if(i==1): c="r"
		ax.plot(x_ticks, ipc_axis, label="Configuration "+str(dw), marker=markers[i%len(markers)],color=c)
		i = i + 1

	lgd = ax.legend(ncol=len(tuples_by_dw), bbox_to_anchor=(0.75, -0.15), prop={'size':8})
	plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')


def lastplotter(t1,t2):
	results_tuples = []
	results_tuples.append((1,1,t1[0]))
	results_tuples.append((1,2,t1[1]))
	results_tuples.append((1,4,t1[2]))
	results_tuples.append((1,8,t1[3]))
	results_tuples.append((1,16,t1[4]))
	results_tuples.append((1,32,t1[5]))
	results_tuples.append((1,64,t1[6]))
	results_tuples.append((2,1,t2[0]))
	results_tuples.append((2,2,t2[1]))
	results_tuples.append((2,4,t2[2]))
	results_tuples.append((2,8,t2[3]))
	results_tuples.append((2,16,t2[4]))
	results_tuples.append((2,32,t2[5]))
	results_tuples.append((2,64,t2[6]))

	return results_tuples


t1 = reader('part11')
t2 = reader('part12')

t3 = reader('part21')
t4 = reader('part22')

print("Done reading files,now let's plot em!")
#print(t1,t2)

# uncomment in order to print line plots
res1 = lastplotter(t1,t2)

printgraphs(res1,'naive_bank.png','Bank accounts 1')

res2 = lastplotter(t3,t4)

printgraphs(res2,'padded_bank.png','Bank accounts 2')







