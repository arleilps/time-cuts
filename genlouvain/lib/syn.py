import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
import pywt
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering

import random

def synthetic_graph(size, num_edges, sparsity, energy, balance, noise):
	size_part_a = int(math.ceil(float(size * balance) / 2))
	size_part_b = size - size_part_a
	F = []
	edges = {}
    
	avg_a = float(numpy.sqrt(float(energy * size) / (size_part_a * size_part_b))) / 2.
    
	avg_b = -float(numpy.sqrt(float(energy * size) / (size_part_a * size_part_b))) / 2.
    
	for v in range(size):
		if v < size_part_a:
			F.append(random.gauss(avg_a, noise*avg_a))
		else:
			F.append(random.gauss(avg_b, noise*avg_a))
    
	G = networkx.Graph()
    
	for v in range(size-1):
		G.add_edge(v,v+1)
		edges[(v,v+1)] = True
        
	remaining_edges = num_edges - len(G.edges())
	edges_accross = int((size_part_a * size_part_b * (1.-sparsity) * remaining_edges) / (size * (size-1)))
	edges_within = remaining_edges - edges_accross
    
	for e in range(edges_accross):
		v1 = random.randint(0, size_part_a-1)
		v2 = random.randint(size_part_a, size-1)
                     
		while (v1,v2) in edges or v1 == v2:
			v1 = random.randint(0,size_part_a-1)
			v2 = random.randint(size_part_a, size_part_a+size_part_b-1)
            
		G.add_edge(v1,v2)
		edges[(v1,v2)] = True
        
	for e in range(edges_within):
		v1 = random.randint(0,size-1)
		v2 = random.randint(0,size-1)
        
		if v1 > v2:
			tmp = v1
			v1 = v2
			v2 = tmp
        
		while (v1,v2) in edges or v1 == v2 or (v1 < size_part_a and v2 >= size_part_a) or (v1 >= size_part_a and v2 < size_part_a):
			v1 = random.randint(0,size-1)
			v2 = random.randint(0,size-1)
            
			if v1 > v2:
				tmp = v1
				v1 = v2
				v2 = tmp
             
		G.add_edge(v1,v2)
		edges[(v1,v2)] = True
        
	return G, numpy.array(F), edges_accross+1

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

def generate_dyn_heat(G, s, jump, n):
	Fs = []
	L = networkx.normalized_laplacian_matrix(G)
	L = L.todense()
	F0s = []	
	seeds = []

	for i in range(s):
		F0 = numpy.zeros(len(G.nodes()))
		v = random.randint(0, len(G.nodes())-1)
		seeds.append(v)
		F0[v] = len(G.nodes())
		F0s.append(F0)

	Fs.append(numpy.sum(F0s, axis=0))

	for j in range(n):
		FIs = []
		for i in range(s):
			FI = numpy.multiply(linalg.expm(-j*jump*L), F0s[i])[:,seeds[i]]
			FIs.append(FI)
		
		Fs.append(numpy.sum(FIs, axis=0))

	return numpy.array(Fs)[1:]

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
