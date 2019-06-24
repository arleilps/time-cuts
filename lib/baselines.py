'''
	Implements several baselines used in the paper.
'''

import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import sparse

from lib.vis import *
from lib.graph_signal_proc import *
from lib.optimal_cut import *
from lib.time_graph import *
from lib.run_estr import *
from lib.run_lovain import *
from lib.run_facetnet import *

def get_assignments(G, labels):
	assign = numpy.zeros(G.num_snaps() * G.size())
	for t in range(G.num_snaps()):
		for v in G.snap(t).nodes():
			i = G.index_vertex(t,v)

			assign[i] = labels[v][t]
	return assign

def write_ncol(G, folder_file_name):
	os.system("rm -r "+folder_file_name)
	os.system("mkdir -p "+folder_file_name)

	node_ids = {}
	ID = 0
	node_ids_rev = []
	
	for t in range(G.num_snaps()):
		out_file_name = folder_file_name +"/"+ str(t+1) + ".ncol"
		out_file = open(out_file_name, 'w')
 
		for e in G.snap(t).edges():
			if e[0] not in node_ids:
				node_ids[e[0]] = ID
				ID = ID + 1
				node_ids_rev.append(e[0])

			if e[1] not in node_ids:
				node_ids[e[1]] = ID
				ID = ID + 1
				node_ids_rev.append(e[1])

			out_file.write(str(node_ids[e[0]])+" "+str(node_ids[e[1]])+" "+str(G.snap(t)[e[0]][e[1]]['weight'])+"\n")
 
	out_file.close()

	return node_ids_rev

def merge_communities(G, assign, K, omega):
	new_assign = numpy.zeros(assign.shape[0])

	part = []
	n = 0

	for i in range(assign.shape[0]):
		while assign[i] >= len(part):
			part.append([])
			n = n + 1

		part[int(assign[i])].append(i)

	M =  create_modularity_matrix(G, omega)

	while n > K:
#		print(n)
#		sys.stdout.flush()
		best_pair = None
		best_score = -sys.float_info.max
		for i in range(len(part)):
			for j in range(len(part)):
				if i < j:
					score = 0.
					for v in part[i]:
						for u in part[j]:
							score = score + 2.*M[v,u]

					if score > best_score:
						best_score = score
						best_pair = (i,j)

		i = best_pair[0]
		j = best_pair[1]
		C = []

		for v in part[i]:
			C.append(v)
		
		for v in part[j]:
			C.append(v)

		if i > j:
			del part[i]
			del part[j]
		else:
			del part[j]
			del part[i]
		
		part.append(C)
		n = n - 1

	return get_partition_assign(G, part)

def facet_net(G, K, lamb):
	node_ids_rev = G.write("graph.tmp")

	labels_int = run_facetnet_method("graph.tmp", K, lamb)

	labels = {}
	for lb in labels_int:
		labels[node_ids_rev[int(lb)]] = labels_int[lb]

	assign = get_assignments(G, labels)

	return assign

def gen_lovain(G, omega, K):
	node_ids_rev = G.write("graph.tmp")
	labels_int = run_lovain_method("graph.tmp", omega)

	labels = {}
	for lb in labels_int:
		labels[node_ids_rev[int(lb)]] = labels_int[lb]


	assign = get_assignments(G, labels)

	return merge_communities(G, assign, K, omega)

def estrangement(G, delta, K, omega):
	node_ids_rev = write_ncol(G, "graph_dir_tmp")
	labels_int = run_estrangement("graph_dir_tmp", delta)

	labels = {}
	for lb in labels_int:
		labels[node_ids_rev[int(lb)]] = labels_int[lb]

	assign = get_assignments(G, labels)

	return merge_communities(G, assign, K, omega)

def estrangement_search(G, K, omega):
	deltas = numpy.arange(.0, 1., 0.1)

	best_score = None
	best_assign = None
	best_delta = None
	for d in deltas:
		assign = estrangement(G, d, K, omega)
		part = get_partitions(G, assign)
		score = evaluate_multi_cut_ratio(G, part, assign)
			
		if best_score is None or score < best_score:
			best_score = score
			best_assign = assign
			best_delta = d
	
	return best_assign, best_delta

def gen_lovain_search(G, K, min_omega=0., max_omega=.1, step=0.01):
	omegas = numpy.arange(7., 8., 10)

	best_score = None
	best_assign = None
	best_omega = None
	for o in omegas:
		assign = gen_lovain(G, o, K)
		part = get_partitions(G, assign)
		score = evaluate_multi_cut_ratio(G, part, assign)
			
		if best_score is None or score < best_score:
			best_score = score
			best_assign = assign
			best_omega = o
	
	return best_assign, best_omega 

def facet_net_search(G, K, omega):
	lambdas = numpy.arange(0., 1., 0.1)

	best_score = None
	best_assign = None
	best_lambda = None
	for lamb in lambdas:
		assign = facet_net(G, K, lamb)
		part = get_partitions(G, assign)
		score = evaluate_multi_cut_ratio(G, part, assign)
			
		if best_score is None or score < best_score:
			best_score = score
			best_assign = assign
			best_lambda = lamb
	
	return best_assign, best_lambda
