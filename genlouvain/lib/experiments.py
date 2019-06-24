import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.netpros import *
from lib.static import *
from mpl_toolkits.mplot3d import axes3d
import numpy
from matplotlib.mlab import griddata
from matplotlib import cm
from lib.syn import *
import time
import sys

def plot_dyn_cut_experiments(results, metric, output_file_name):
	plt.clf()
	ncol=2
	ax = plt.subplot(111)
	width = 0.5       # the width of the bars
	
	i = 0
	labels = []
	if "TP" in results:
		ax.bar([i], results["TP"][metric], width, color='cyan', label="TP", hatch="/")
		i = i + 1
		labels.append('TP')
	
	if "TI" in results: 
		ax.bar([i], results["TI"][metric], width, color='cyan', label="TI", hatch="\\")
		i = i + 1
		labels.append('TI')
	
	if "TD" in results: 
		ax.bar([i], results["TD"][metric], width, color='cyan', label="TD", hatch="-")
		i = i + 1
		labels.append('TD')
	
	if "ID" in results: 
		ax.bar([i], results["ID"][metric], width, color='darkgreen', label="ID", hatch="*")
		i = i + 1
		labels.append('ID')

	if "IC" in results: 
		ax.bar([i], results["IC"][metric], width, color='darkgreen', label="IC", hatch="/")
		i = i + 1
		labels.append('IC')

	if "DC" in results: 
		ax.bar([i], results["DC"][metric], width, color='k', label="DC", hatch="\\")
		i = i + 1
		labels.append('DC')
	
	if "UC" in results: 
		ax.bar([i], results["UC"][metric], width, color='k', label="UC", hatch="-")
		i = i + 1
		labels.append('UC')
	
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.set_ylabel(metric, fontsize=30)
	ax.set_xlabel('Algorithm', fontsize=30)
	ax.set_xlim([-0.5,i])
	ax.set_xticks(numpy.arange(i) + width/2)
	ax.set_xticklabels(labels)
	
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def dyn_cut_experiments(G, algs, n=1):
	results = {}

	for alg in algs:
		results[alg.name()] = {}
		results[alg.name()]["time"] = []
		results[alg.name()]["ratio"] = []
		results[alg.name()]["cost"] = []
			
		for i in range(n):
			
			start_time = time.time()

			c = alg.cut(G)

			results[alg.name()]["ratio"] = c["score"]
			results[alg.name()]["cost"] = c["edges"]
			
			t = time.time()-start_time

			results[alg.name()]["time"].append(t)

		results[alg.name()]["time"] = numpy.array(results[alg.name()]["time"]) 
		results[alg.name()]["time"] = numpy.mean(results[alg.name()]["time"])
		
		results[alg.name()]["ratio"] = numpy.array(results[alg.name()]["ratio"]) 
		results[alg.name()]["ratio"] = numpy.mean(results[alg.name()]["ratio"])
		
		results[alg.name()]["cost"] = numpy.mean(results[alg.name()]["cost"])
		results[alg.name()]["cost"] = numpy.mean(results[alg.name()]["cost"])
              
	return results

