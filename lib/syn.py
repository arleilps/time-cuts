import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import matplotlib.pyplot as plt
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from lib.time_graph import *

import random

def slide_diagonal_grids(n, k, W):
	Grids = []
	r = 0
	c = 0
	i = 0
	while r <=  n-k and c <= n-k:
		Grids.append(numpy.zeros((n,n)))
		Grids[-1][r:r+k, c:c+k] = W*numpy.ones((k,k))
		
		if i % 2 == 0:
			r = r + 1
		else:
			c = c + 1
		i = i + 1
		
	return numpy.array(Grids)

def grow_diagonal_grids(n, k, W):
	Grids = []
	r = 0
	c = 0
	i = 0
	while r <=  n-k and c <= n-k:
		Grids.append(numpy.zeros((n,n)))
		Grids[-1][0:r+k, 0:c+k] = W*numpy.ones((r+k,c+k))
		
		if i % 2 == 0:
			r = r + 1
		else:
			c = c + 1
		i = i + 1
		
	return numpy.array(Grids)

def add_gaussian_noise(grids, s):
	new_grids = numpy.copy(grids)
	for g in range(grids.shape[0]):
		new_grids[g] = grids[g] + numpy.random.normal(0, s, (grids.shape[1],grids.shape[2]))
		
	return new_grids

def time_graph_from_grid(grid, n_hops):
	G = TimeGraph(1.)
	values = {}
	for t in range(grid.shape[0]):
		values[t] = {}
		for i in range(grid.shape[1]):
			for j in range(grid.shape[2]):
				values[t][(i,j)] = grid[t][i,j]
				
				for hi in range(grid.shape[1]):
					for hj in range(grid.shape[2]):
						if (i,j) != (hi,hj) and numpy.absolute(i - hi) <= n_hops and numpy.absolute(j - hj) <= n_hops:
								v1 = (i,j)
								v2 = (hi,hj)
								w = numpy.exp(-numpy.absolute(grid[t][i,j]-grid[t][hi,hj]))

								G.add_edge(v1, v2, t, w)
	G.index_vertex()

	for t in range(grid.shape[0]-1):
		for i in range(grid.shape[1]):
			for j in range(grid.shape[2]):
				v = (i,j)
				w = numpy.exp(-numpy.absolute(grid[t][i,j]-grid[t+1][i,j]))

				G.set_swap_cost_vertex(v, t, w)
	return G, values		   

def downsample(grid, rate):
	new_grid = numpy.copy(grid)
	k = int(math.ceil(1. / rate))

	for i in range(grid.shape[0]):
		if i % k != 0:
			new_grid[i] = new_grid[i-1]

	return new_grid

def compute_distances(center, graph):
	distances = networkx.shortest_path_length(graph, center)
	
	return distances

def compute_embedding(distances, radius, graph):
	B = []
	s = 0
	nodes = {}
	for v in graph.nodes():
		if distances[v] <= radius:
			B.append(1)
			s = s + 1
		else:
			B.append(0)
			
	return numpy.array(B)

def generate_dyn_cascade(G, diam, duration, n):
	Fs = []
	
	for j in range(n):
		v = random.randint(0, len(G.nodes())-1)
		distances = compute_distances(G.nodes()[v], G)

		if diam > duration:
			num_snaps = diam
		else:
			num_snaps = duration
	 
		for i in range(num_snaps):
			r = int(i * math.ceil(float(diam)/duration))
		
			F = compute_embedding(distances, r, G)
			Fs.append(F)
		
	return numpy.array(Fs)

def generate_dyn_heat(G, s, jump):
	Fs = []
	seeds = []
	F0s = []	
	G.index_vertex()
	for i in range(s):
		F0 = numpy.zeros(len(G.nodes()))
		v = random.randint(0, len(G.nodes())-1)
		seeds.append(v)
		F0[v] = len(G.nodes())
		F0s.append(F0)

	Fs.append(numpy.sum(F0s, axis=0))

	for j in range(G.num_snaps()-1):
		FIs = []
		L = networkx.normalized_laplacian_matrix(G.snap(j), G.nodes(), weight='weight')
		L = L.todense()
		L = numpy.array(L)
		for i in range(len(G.nodes())):
			FI = numpy.multiply(linalg.expm(-jump*L), Fs[-1][i])[:,i]
			FIs.append(FI)
		
		Fs.append(numpy.sum(FIs, axis=0))
		Fs[-1] = (Fs[-1] / max(Fs[-1])) * len(G.nodes())

	return numpy.array(Fs)

def generate_dyn_gaussian_noise(G, n):
	Fs = []
	
	for j in range(n):
		F = numpy.random.rand(len(G.nodes()))
		Fs.append(F)

	return numpy.array(Fs)

def generate_dyn_bursty_noise(G, n):
	Fs = []
	bursty_beta = 1
	non_bursty_beta = 1000
	bursty_bursty = 0.7
	non_bursty_non_bursty = 0.9
	bursty = False

	for j in range(n):
		r = random.random()

		if not bursty:
			if r > non_bursty_non_bursty:
				bursty = True
		else:
			if r > bursty_bursty:
				bursty = False

		if bursty:	
			F = numpy.random.exponential(bursty_beta, len(G.nodes()))
		else:
			F = numpy.random.exponential(non_bursty_beta, len(G.nodes()))
			
		Fs.append(F)

	return numpy.array(Fs)

def generate_dyn_indep_cascade(G, s, p):
	Fs = []
	
	seeds = numpy.random.choice(len(G.nodes()), s, replace=False)
	
	F0 = numpy.zeros(len(G.nodes()))
	
	ind = {}
	i = 0

	for v in G.nodes():
		ind[v] = i
		i = i + 1
	
	for s in seeds:
		F0[s] = 2.0

	while True:
		F1 = numpy.zeros(len(G.nodes()))
		new_inf = 0
		for v in G.nodes():
			if F0[ind[v]] > 1.0:
				for u in G.neighbors(v):
					r = random.random()
					if r <= p and F0[ind[u]] < 1.0:
						F1[ind[u]] = 2.0
						new_inf = new_inf + 1
				F1[ind[v]] = 1.0
				F0[ind[v]] = 1.0
			elif F0[ind[v]] > 0.0:
				F1[ind[v]] = 1.0
		
		Fs.append(F0)
		
		if new_inf == 0 and len(Fs) > 1:
			break

		F0 = numpy.copy(F1)
	
	return numpy.array(Fs)

def generate_dyn_linear_threshold(G, s):
	Fs = []
	
	seeds = numpy.random.choice(len(G.nodes()), s, replace=False)
	
	F0 = numpy.zeros(len(G.nodes()))
	thresholds = numpy.random.uniform(0.0,1.0,len(G.nodes()))
	
	ind = {}
	i = 0

	for v in G.nodes():
		ind[v] = i
		i = i + 1
	
	for s in seeds:
		F0[s] = 1.0

	while True:
		F1 = numpy.zeros(len(G.nodes()))
		new_inf = 0
		for v in G.nodes():
			if F0[ind[v]] < 1.0:
				n = 0			
				for u in G.neighbors(v):
					if F0[ind[u]] > 0:
						n = n + 1
				
				if (float(n) / len(G.neighbors(v))) >= thresholds[ind[v]]:
					F1[ind[v]] = 1.0
					new_inf = new_inf + 1
			else:
				F1[ind[v]] = 1.0					
	
		Fs.append(F0)
		
		if new_inf == 0 and len(Fs) > 1:
			break

		F0 = numpy.copy(F1)
	
	return numpy.array(Fs)
