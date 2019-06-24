import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lib.syn import *
from lib.time_graph import *
from lib.baselines import *
from mpl_toolkits.mplot3d import axes3d
import numpy
from matplotlib.mlab import griddata
from matplotlib import cm
from lib.syn import *
from lib.time_graph import *
from lib.time_graph_signal_proc import *
import time
import sys
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

def plot_fast_sparsity_experiments(results, algs, swap_cost, output_file_name, xmin=None,xmax=None, ymin=None,ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars
	colors = ["green", "blue", "gray", "magenta"]
	i = 0
	j = -6
	x = numpy.arange(100, 100*len(swap_cost)+1, 100)
	for a in algs:
		label = a.name()
		y = []

		for r in results:
			y.append(r[i]["sparsity"])

		ax.bar(x+(i+j)*width/2, numpy.array(y), width, color=colors[i], label=label)

		if j == 6:
			j = -6
		else:
			j = j + 1
		
		i = i + 1

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper center", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('ratio', fontsize=30)
	ax.set_xlabel('swap cost', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_yscale('log')
	ax.set_xticks(x)
	ax.set_xticklabels(swap_cost)
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_sparsity_experiments(results, algs, swap_cost, output_file_name, xmin=None,xmax=None, ymin=None,ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars
	colors = ["green", "blue","red", "black", "gray", "magenta"]
	i = 0
	j = -6
	x = numpy.arange(100, 100*len(swap_cost)+1, 100)
	for a in algs:
		label = a.name()
		y = []

		for r in results:
			y.append(r[i]["sparsity"])

		ax.bar(x+(i+j)*width/2, numpy.array(y), width, color=colors[i], label=label)

		if j == 6:
			j = -6
		else:
			j = j + 1
		
		i = i + 1

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper center", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('ratio', fontsize=30)
	ax.set_xlabel('swap cost', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_yscale('log')
	ax.set_xticks(x)
	ax.set_xticklabels(swap_cost)
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_rank_sparsity_experiments(results, rank, output_file_name, xmin=None,xmax=None, ymin=None,ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 0.5       # the width of the bars
	
	markers = ["*", "o"]
	colors = ["red", "green"]
	i = 0
	
	y = []
	for r in results:
		y.append(r[0]["sparsity"])

	ax.plot(numpy.array(rank), numpy.array(y), marker=markers[0], color=colors[0], label="STC", markersize=20)
	
	y = []
	for r in results:
		y.append(r[1]["sparsity"])

	ax.plot(numpy.array(rank), numpy.array(y), marker=markers[1], color=colors[1], label="FSTC", markersize=20)
		
	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper center", prop={'size':15}, ncol=4, numpoints=1)
	ax.set_ylabel('ratio', fontsize=30)
	ax.set_xlabel('rank', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_ylim(ymin, ymax)
	ax.set_xlim(xmin, xmax)
	ax.set_xscale("log", basex=2)
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_size_experiments(results, algs, size, output_file_name, xmin=None,xmax=None, ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars
	colors = ["green", "blue","red", "black", "gray", "magenta"]
	i = 0
	j = -6
	x = numpy.arange(100, 100*len(size)+1, 100)
	for a in algs:
		label = a.name()
		y = []

		for r in results:
			y.append(r[i]["time"])

		ax.bar(x+(i+j)*width/2, numpy.array(y), width, color=colors[i], label=label)

		if j == 6:
			j = -6
		else:
			j = j + 1
		
		i = i + 1

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper left", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('size graph', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	if ymax is not None:
		ax.set_ylim(1.,ymax)
	ax.set_xticks(x)
	ax.set_xticklabels(size)
	ax.set_yscale('log', nonposy='clip')
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_hop_experiments(results, algs, hop, output_file_name, xmin=None,xmax=None, ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars
	colors = ["green", "blue","red", "black", "gray", "magenta"]
	i = 0
	j = -6
	x = numpy.arange(100, 100*len(hop)+1, 100)
	for a in algs:
		label = a.name()
		y = []

		for r in results:
			y.append(r[i]["time"])

		ax.bar(x+(i+j)*width/2, numpy.array(y), width, color=colors[i], label=label)

		if j == 6:
			j = -6
		else:
			j = j + 1
		
		i = i + 1

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper left", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('density', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	
	if ymax is not None:
		ax.set_ylim(1.,ymax)

	ax.set_xticks(x)
	ax.set_xticklabels(hop)
	ax.set_yscale('log')
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def plot_num_snaps_experiments(results, algs, num_snaps, output_file_name, xmin=None,xmax=None, ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars
	colors = ["green", "blue","red", "black", "gray", "magenta"]
	i = 0
	j = -6
	x = numpy.arange(100, 100*len(num_snaps)+1, 100)
	for a in algs:
		label = a.name()
		y = []

		for r in results:
			y.append(r[i]["time"])

		ax.bar(x+(i+j)*width/2, numpy.array(y), width, color=colors[i], label=label)

		if j == 6:
			j = -6
		else:
			j = j + 1
		
		i = i + 1

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper center", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('#snapshots', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	ax.set_xticks(x)
	ax.set_xticklabels(num_snaps)
	
	if ymax is not None:
		ax.set_ylim(1., ymax)
	
	ax.set_yscale('log')
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def plot_rank_time_experiments(results, rank, output_file_name, xmin=None,xmax=None,ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 40       # the width of the bars
	colors = ["red", "green"]
	x = numpy.arange(100, 100*len(rank)+1, 100)

	y = []
	for r in results:
		y.append(r[0]["time"])

	ax.bar(x+(-2)*width/2, numpy.array(y), width, color=colors[0], label="STC")
	
	y = []
	for r in results:
		y.append(r[1]["time"])

	ax.bar(x, numpy.array(y), width, color=colors[1], label="FSTC")

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="upper left", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel('time (sec.)', fontsize=30)
	ax.set_xlabel('rank', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	
	if ymax is not None:
		ax.set_ylim(1., ymax)
	
	ax.set_xticks(x)
	ax.set_xticklabels(rank)
	ax.set_yscale('log')
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def sparsity_experiments(G, algs, swp_costs,  n=1):
	results = []
	
	for p in range(len(swp_costs)):
		results.append([])

		G.index_vertex()
		G.set_swap_cost(swp_costs[p])

		for alg in algs:
			sparsity = numpy.zeros(n)
			times = numpy.zeros(n)

			for i in range(n):
				start_time = time.time()
				c = alg.cut(G)
				sparsity[i] = c["score"]

				t = time.time()-start_time
				times[i] = t

			s = numpy.mean(sparsity)
			t = numpy.mean(times)

			res = {}
			res["time"] = t
			res["sparsity"] = s

			results[-1].append(res)
	
	return results

def rank_experiments(G, ranks, n=1, sparse=True):
	results = []

	G.index_vertex()
	if sparse is True:
		alg = TemporalCuts("STC", "diff-sparse", 1e-5)	
	else:
		alg = TemporalCuts("STC", "diff-norm", 1e-5)	
	
	res_c = {}
	sparsity = numpy.zeros(n)
	times = numpy.zeros(n)
		
	for i in range(n):
		start_time = time.time()
		c = alg.cut(G)
		sparsity[i] = c["score"]
	
		t = time.time()-start_time
		times[i] = t

	s = numpy.mean(sparsity)
	t = numpy.mean(times)


	res_c["time"] = t
	res_c["sparsity"] = s

	for r in range(len(ranks)):
		results.append([])
	
		sparsity = numpy.zeros(n)
		times = numpy.zeros(n)
		if sparse is True:
			alg = TemporalCuts("FSTC", "fast-sparse", 1e-5, k=ranks[r])	
		else:
			alg = TemporalCuts("FSTC", "fast-norm", 1e-5, k=ranks[r])	

		for i in range(n):
			start_time = time.time()
			alg.set_rank(ranks[r])
			c = alg.cut(G)
			sparsity[i] = c["score"]
	
			t = time.time()-start_time
			times[i] = t

		s = numpy.mean(sparsity)
		t = numpy.mean(times)

		results[-1].append(res_c)
		
		res = {}
		res["time"] = t
		res["sparsity"] = s
		results[-1].append(res)
	
	return results

def size_experiments(algs, size, slide=True, n=1):
	results = []
	
	data = []
	num_snaps = None
	for z in size:
		
		if slide is True:
			grid = slide_diagonal_grids(int(math.ceil(math.sqrt(z))), int(math.ceil(math.sqrt(z/2))), 1.)
		else:
			grid = grow_diagonal_grids(int(math.ceil(math.sqrt(z))), int(math.ceil(math.sqrt(z/2))), 1.)
		
		if num_snaps is None:
			num_snaps = grid.shape[0]
		else:
			grid = grid[0:num_snaps]
		
		grid = add_gaussian_noise(grid, .5)
		G, values = time_graph_from_grid(grid, 1)
		data.append(G)
	
	for z in range(len(size)):
		results.append([])
		for alg in algs:
			sparsity = numpy.zeros(n)
			times = numpy.zeros(n)

			for i in range(n):
				start_time = time.time()
				c = alg.cut(data[z])
				sparsity[i] = c["score"]

				t = time.time()-start_time
				times[i] = t

			s = numpy.mean(sparsity)
			t = numpy.mean(times)

			res = {}
			res["time"] = t
			res["sparsity"] = s

			results[-1].append(res)
	
	return results

def hop_experiments(algs, hop, slide=True, n=1):
	results = []
	
	data = []
	for h in hop:
		if slide is True:
			grid = slide_diagonal_grids(30, 15, 1.)
		else:
			grid = grow_diagonal_grids(30, 15, 1.)
		
		grid = add_gaussian_noise(grid, .5)
		G, values = time_graph_from_grid(grid, h)
		data.append(G)
	
	for h in range(len(hop)):
		results.append([])
		for alg in algs:
			sparsity = numpy.zeros(n)
			times = numpy.zeros(n)

			for i in range(n):
				start_time = time.time()
				c = alg.cut(data[h])
				sparsity[i] = c["score"]

				t = time.time()-start_time
				times[i] = t

			s = numpy.mean(sparsity)
			t = numpy.mean(times)

			res = {}
			res["time"] = t
			res["sparsity"] = s

			results[-1].append(res)
	
	return results


def plot_compression_experiments(results, K, output_file_name, xmin=None,xmax=None, ymin=None,ymax=None):
	plt.clf()
	ax = plt.subplot(111)
	width = 15       # the width of the bars

	ax.plot(numpy.power(2, numpy.arange(len(K))), results[0], marker="o", color="green", label="Fourier", markersize=15)
	ax.plot(numpy.power(2, numpy.arange(len(K))), results[1], marker="*", color="blue", label="STC-0", markersize=15)
	ax.plot(numpy.power(2, numpy.arange(len(K))), results[2], marker="s", color="red", label="STC-100", markersize=15)
	ax.plot(numpy.power(2, numpy.arange(len(K))), results[3], marker="v", color="black", label="STC-200", markersize=15)

	plt.gcf().subplots_adjust(bottom=0.15)
	ax.legend(loc="lower center", prop={'size':15}, ncol=3, numpoints=1)
	ax.set_ylabel(r'L$_2$', fontsize=30)
	ax.set_xlabel('k', fontsize=30)
	ax.tick_params(labelsize=23)
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(ymin, ymax)
	ax.set_yscale('log', basey=2)
	ax.set_xscale('log',basex=2)
	ax.set_xticks(numpy.power(2, numpy.arange(len(K))))
	ax.set_xticklabels(K)
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')


def compression_experiments(G, F, K, n):
	results = numpy.zeros((4, len(K)))

	for i in range(len(K)):
		k = K[i]

		avg = 0.
		for j in range(n):
			f = graph_fourier_transform(G, F, k)		
			e = L2_error(F, f)
			avg = avg + e
		avg = avg/n

		results[0][i] = avg

		avg = 0.
		for j in range(n):
			c = temporal_graph_transform(G, F, k, alpha=0., order=1.)		
			f = c["transform"]
			e = L2_error(F, f)
			avg = avg + e
		avg = avg/n
		
		results[1][i] = avg
		
		avg = 0.
		for j in range(n):
			c = temporal_graph_transform(G, F, k, alpha=100., order=1.)		
			f = c["transform"]
			e = L2_error(F, f)
			avg = avg + e
		avg = avg/n
		
		results[2][i] = avg
		
		avg = 0.
		for j in range(n):
			c = temporal_graph_transform(G, F, k, alpha=200., order=1.)		
			f = c["transform"]
			e = L2_error(F, f)
			avg = avg + e
		avg = avg/n

		results[3][i] = avg

	return results


def num_snaps_experiments(algs, num_snaps, slide=True, n=1):
	results = []
	
	data = []
	for u in num_snaps:
		if slide is True:
			grid = slide_diagonal_grids(40, 20, 2.)
		else:
			grid = grow_diagonal_grids(40, 20, 2.)

		grid = add_gaussian_noise(grid, 1.)
		grid = grid[:u]

		G, values = time_graph_from_grid(grid, 2)
		data.append(G)
	
	for u in range(len(num_snaps)):
		results.append([])
		for alg in algs:
			sparsity = numpy.zeros(n)
			times = numpy.zeros(n)

			for i in range(n):
				start_time = time.time()
				c = alg.cut(data[u])
				sparsity[i] = c["score"]

				t = time.time()-start_time
				times[i] = t

			s = numpy.mean(sparsity)
			t = numpy.mean(times)

			res = {}
			res["time"] = t
			res["sparsity"] = s

			results[-1].append(res)
	
	return results

def plot_community_detection_experiments(results, fun, output_file_name):
	plt.clf()
	ncol=5
	ax = plt.subplot(111)
	width = 0.8       # the width of the bars
	ax.bar([1-width/2], results["gen_lovain"][fun], width, color='gray')
	ax.bar([2-width/2], results["estrangement"][fun], width, color='black')
	ax.bar([3-width/2], results["facetnet"][fun], width, color='pink')
	ax.bar([4-width/2], results["stc"][fun], width, color='red')
	ax.bar([5-width/2], results["nstc"][fun], width, color='green')

	plt.gcf().subplots_adjust(bottom=0.15)
#	ax.legend(loc='upper center', prop={'size':20}, ncol=ncol)
	ax.legend_ = None
	ax.set_ylabel(fun, fontsize=30)
#	ax.set_xlabel('Methods', fontsize=30)
	labels = ['GLO', 'EST', 'FCN', 'STC', 'NTC']
	plt.xticks([1, 2, 3, 4, 5], labels)
	ax.set_xlim(0, 6)
	plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	plt.rcParams['xtick.labelsize'] = 30 
	plt.rcParams['ytick.labelsize'] = 20
	plt.savefig(output_file_name, dpi=300, bbox_inches='tight')

def community_detecion_experiments(G, K, n=1):
	avg_lovain = None
	avg_fn = None
	avg_stc = None
	avg_nstc = None
	for i in range(n):
		(lovain_assign, lovain_omega) = gen_lovain_search(G, K)
		part = get_partitions(G, lovain_assign)
		lovain = evaluate_multi_cut(G, part, lovain_assign, lovain_omega)
		
		if avg_lovain is None:
			avg_lovain = lovain
		else:
			for k in avg_lovain:
				avg_lovain[k] = avg_lovain[k] + lovain[k]

		(facetnet_assign, facetnet_lambda) = facet_net_search(G, K, lovain_omega)
		part = get_partitions(G, facetnet_assign)
		fn = evaluate_multi_cut(G, part, facetnet_assign, lovain_omega)
		
		if avg_fn is None:
			avg_fn = fn
		else:
			for k in avg_fn:
				avg_fn[k] = avg_fn[k] + fn[k]

		if K == 2:
			stc_assign = multi_cut_hierarchy(G, K)
		else:
			stc_assign = multi_cut_kmeans(G, K)
	
		part = get_partitions(G, stc_assign)
		stc = evaluate_multi_cut(G, part, stc_assign, lovain_omega)
		
		if avg_stc is None:
			avg_stc = stc
		else:
			for k in avg_stc:
				avg_stc[k] = avg_stc[k] + stc[k]
	
		if K == 2:
			nstc_assign = multi_cut_hierarchy(G, K, True)
		else:
			nstc_assign = multi_cut_kmeans(G, K, True)

		part = get_partitions(G, nstc_assign)
		nstc = evaluate_multi_cut(G, part, nstc_assign, lovain_omega)

		if avg_nstc is None:
			avg_nstc = nstc
		else:
			for k in avg_nstc:
				avg_nstc[k] = avg_nstc[k] + nstc[k]
	
	res = {}

	for k in avg_fn:
		avg_fn[k] = avg_fn[k]/n
	
	for k in avg_lovain:
		avg_lovain[k] = avg_lovain[k]/n
	
	for k in avg_stc:
		avg_stc[k] = avg_stc[k]/n
	
	for k in avg_nstc:
		avg_nstc[k] = avg_nstc[k]/n

	res["gen_lovain"] = avg_lovain
	res["facetnet"] = avg_fn
	res["stc"] = avg_stc
	res["nstc"] = avg_nstc

	return res
