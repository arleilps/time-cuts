import networkx
import math
import numpy
import scipy
import sys
from scipy import linalg
import time
from numpy import log, mean, dot, diag, sqrt
from numpy.linalg import eigh
from lib.graph_signal_proc import *
from scipy import sparse

def sqrtm(mat):
	"""
	Matrix square root.
	@type  mat: array_like
	@param mat: matrix for which to compute square root
	"""

	# find eigenvectors
	eigvals, eigvecs = eigh(mat)
	
	# eliminate eigenvectors whose eigenvalues are zero
	eigvecs = eigvecs[:, eigvals > 0]
	eigvals = eigvals[eigvals > 0]
	
	# matrix square root
	return dot(eigvecs, dot(diag(sqrt(eigvals)), eigvecs.T))


def sweep_vec(x, beta, F, G, k, ind):
	best_val = 0.
	best_edges_cut = 0
	sorted_x = numpy.argsort(x)
	size_one = 0
	sum_one = 0
	sum_two = 0
	
	for v in G.nodes():
		sum_two = sum_two + F[ind[v]]
	
	edges_cut = 0 
	nodes_one = {}
	total_size = networkx.number_of_nodes(G)

	for i in range(x.shape[0]):
		size_one = size_one + 1
		sum_one = sum_one + F[ind[G.nodes()[sorted_x[i]]]]
		sum_two = sum_two - F[ind[G.nodes()[sorted_x[i]]]]
		
		nodes_one[G.nodes()[sorted_x[i]]] = True
		
		for v in G.neighbors(G.nodes()[sorted_x[i]]):
			if v not in nodes_one:
				edges_cut = edges_cut + 1
			else:
				edges_cut = edges_cut - 1
		
		den = size_one  * (total_size-size_one) * total_size
		if den > 0:
			val = math.pow(sum_one*(total_size-size_one) - sum_two*size_one, 2) / den
		else:
			val = 0
		
		if val >= best_val and edges_cut <= k:
			best_cand = i
			best_val = val
			best_edges_cut = edges_cut
			
			if total_size * size_one * (total_size-size_one) > 0:
				energy = math.pow(sum_one*(total_size-size_one) - sum_two*size_one, 2) / (total_size * size_one * (total_size-size_one))
			else:
				energy = 0

	vec = numpy.zeros(total_size)
	
	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.
	
	return vec, best_val, best_edges_cut, energy

def norm_cut(G,F):
	c = normalized_cut(G)
	
	res = {}
	res["x"] = c

	size_one = 0
	sum_one = 0

	for i in range(c.shape[0]):
		if c[i] < 0:
			size_one = size_one + 1
			sum_one = sum_one + 1

	res["score"] = score
	res["energy"] = energy
	res["size"] = 0

def laplacian_complete(n):
	C = numpy.ones((n, n))
	C = -1 * C
	D = numpy.diag(numpy.ones(n))
	C = (n)*D + C
	
	return C

def weighted_adjacency_complete(G, F, ind):
	A = []
	for v in G.nodes():
		A.append([])
		for u in G.nodes():
			A[-1].append(pow(F[ind[v]]-F[ind[u]],2))
			
	return numpy.array(A)

def fast_cac(G, F, ind):
	CAC = []
	for v in G.nodes():
		CAC.append([])
		for u in G.nodes():
			#CAC[-1].append(math.pow(F[ind[v]] * F[ind[u]], 2))
			CAC[-1].append(F[ind[v]] * F[ind[u]])

	CAC = numpy.array(CAC)
	CAC = -2 * math.pow(networkx.number_of_nodes(G), 2) * CAC

	return CAC

def power_method(mat, start, maxit):
	vec = numpy.copy(start)
	vec = vec/numpy.linalg.norm(vec)

	for i in range(maxit):
		vec = numpy.dot(vec, mat)

	return vec


def spectral_cut(CAC, L, C, A, start, F, G, beta, k, ind):
	isqrtCL = sqrtmi( C + beta * L)
	M = numpy.dot(numpy.dot(isqrtCL, CAC), isqrtCL)
	
	(eigvals, eigvecs) = scipy.linalg.eigh(M,eigvals=(0,0))
	x = numpy.asarray(numpy.dot(eigvecs[:,0], isqrtCL))[0,:]

	(x, score, size, energy) = sweep_vec(x, beta, F, G, k, ind)
	
	res = {}
	res["x"] = numpy.array(x)
	res["size"] = size
	res["score"] = score
	res["energy"] = energy
	
	return res

def eig_vis_opt(G, F, beta):
	ind = {}
	i = 0
	
	for v in G.nodes():
		ind[v] = i
		i = i + 1
	
	C = laplacian_complete(networkx.number_of_nodes(G))
	A = weighted_adjacency_complete(G, F, ind)
	CAC = numpy.dot(numpy.dot(C,A), C)
	L = networkx.laplacian_matrix(G).todense()
	
	isqrtCL = sqrtmi( C + beta * L)
	M = numpy.dot(numpy.dot(isqrtCL, CAC), isqrtCL)
	
	(eigvals, eigvecs) = scipy.linalg.eigh(M,eigvals=(0,1))
	x1 = numpy.asarray(numpy.dot(eigvecs[:,0], isqrtCL))[0,:]
	x2 = numpy.asarray(numpy.dot(eigvecs[:,1], isqrtCL))[0,:]

	return x1, x2

def trans(L, min_v, max_v):
	return (float(2.) / (max_v-min_v)) * L, -(float(max_v+min_v) / (max_v-min_v)) 
    
def fun(k, n, beta, min_v, max_v, x):
	y = 0.5 * math.cos(x) * float(max_v - min_v) +  (0.5 * (max_v + min_v))
	
	#return math.cos(k*x)*(float(1.) / math.sqrt(n + beta*y))
	return math.cos(k*x)*(float(1.) / math.sqrt(beta*y))
		    
def coef(k, n, beta, min_v, max_v):
	return float(2. * scipy.integrate.quad(lambda x: fun(k, n, beta, min_v, max_v, x), 0., math.pi)[0]) / math.pi
			
def chebyshev_approx_2d(n, beta, X, L):
	max_v = beta * L.shape[0]
	min_v = 1
	
	ts1, ts2 = trans(L, min_v, max_v)
	P1 = 0.5 * coef(0, L.shape[0], beta, min_v, max_v) * X
	tkm2 = X 
	tkm1 = scipy.sparse.csr_matrix.dot(ts1, X) + ts2 * X
	P1 = P1 + coef(1, L.shape[0], beta, min_v, max_v) * tkm1
	
	for i in range(2, n):                 
		Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, tkm1) + ts2 * tkm1) - tkm2  
		P1 = P1 + coef(i, L.shape[0], beta, min_v, max_v) * Tk
		tkm2 = tkm1
		tkm1 = Tk
	
	P1 = P1.transpose()    
	P2 = 0.5 * coef(0, L.shape[0], beta, min_v, max_v) * P1
	tkm2 = P1 
	tkm1 = scipy.sparse.csr_matrix.dot(ts1, P1) + ts2 * P1

	P2 = P2 + coef(1, L.shape[0], beta, min_v, max_v) * tkm1
										        
	for i in range(2, n):                 
		Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, tkm1) + ts2 * tkm1) - tkm2  
		P2 = P2 + coef(i, L.shape[0], beta, min_v, max_v) * Tk
		tkm2 = tkm1
		tkm1 = Tk

	return P2

def chebyshev_approx_1d(n, beta, x, L):
	max_v = beta * L.shape[0]
	min_v = 1
	
	ts1, ts2 = trans(L, min_v, max_v)
	P = 0.5 * coef(0, L.shape[0], beta, min_v, max_v) * x
	tkm2 = x
	tkm1 = scipy.sparse.csr_matrix.dot(ts1, x) + ts2 * x
	P = P + coef(1, L.shape[0], beta, min_v, max_v) * tkm1
	
	for i in range(2, n):                 
		Tk = 2. * (scipy.sparse.csr_matrix.dot(ts1, tkm1) + ts2 * tkm1) - tkm2  
		P = P + coef(i, L.shape[0], beta, min_v, max_v) * Tk
		tkm2 = tkm1
		tkm1 = Tk
	
	return P

def simple_spectral_cut(CAC, start, F, G, beta, k, n, ind):
	L = networkx.laplacian_matrix(G)
	M = chebyshev_approx_2d(n, beta, CAC, L)
	
	eigvec = power_method(-M, start, 10)
	x = chebyshev_approx_1d(n, beta, eigvec, L)
	
	(x, score, size, energy) = sweep_vec(x, beta, F, G, k, ind)
	
	res = {}
	res["x"] = numpy.array(x)
	res["size"] = size
	res["score"] = score
	res["energy"] = energy
	
	return res

def fast_search(G, F, k, n, ind):
	start = numpy.ones(networkx.number_of_nodes(G))
	C = laplacian_complete(networkx.number_of_nodes(G))
	A = weighted_adjacency_complete(G, F, ind)
	CAC = fast_cac(G, F, ind)

	return simple_spectral_cut(CAC, start, F, G, 1., k, n, ind)

gr=(math.sqrt(5)-1)/2

def one_d_search(G, F, k, ind):
	C = laplacian_complete(networkx.number_of_nodes(G))
	A = weighted_adjacency_complete(G,F, ind)
	CAC = numpy.dot(numpy.dot(C,A), C)
	start = F
	L = networkx.laplacian_matrix(G).todense()

	a = 0.
	b = 100.
	c=b-gr*(b-a)
	d=a+gr*(b-a)
	tol = 1.
	resab = {}
	resab["size"] = k + 1
	
	while abs(c-d)>tol or resab["size"] > k:      
		resc = spectral_cut(CAC, L, C, A, start, F, G, c, k, ind)
		resd = spectral_cut(CAC, L, C, A, start, F, G, d, k, ind)
		
		if resc["size"] <= k: 
			if resc["score"] > resd["score"]: 
				start = numpy.array(resc["x"])
				b = d
				d = c
				c=b-gr*(b-a)
			else:
				start = numpy.array(resd["x"])
				a=c
				c=d  
				d=a+gr*(b-a)
		else:
				start = numpy.array(resc["x"])
				a=c
				c=d  
				d=a+gr*(b-a)
		
		resab = spectral_cut(CAC, L, C, A, start, F, G, (b+a) / 2, k, ind)
	
	return resab

def get_subgraphs(G, cut):
	G1 = networkx.Graph()
	G2 = networkx.Graph()
	i = 0
	P1 = []
	P2 = []
	for v in G.nodes():
		if cut[i] < 0:
			P1.append(v)
		else:
			P2.append(v)
		i = i + 1

	G1 = G.subgraph(P1)
	G2 = G.subgraph(P2)
	
	return G1, G2

def optimal_wavelet_basis(G, F, k, npol):
	i = 0
	
	ind = {}
	i = 0
	for v in G.nodes():
		ind[v] = i
		i = i+1

	root = Node(None)
	size = 0
	cand_cuts = []

	if npol == 0:
		c = one_d_search(G, F, k, ind)
	else:
		c = fast_search(G, F, k, npol, ind)

	c["parent"] = root
	c["graph"] = G
	
	cand_cuts.append(c)
	while size <= k and len(cand_cuts) > 0:
		best_cut = None
		b = 0
		
		for i in range(0, len(cand_cuts)):
			if cand_cuts[i]["size"] + size <= k and cand_cuts[i]["score"] > 0:
				if best_cut is None or cand_cuts[i]["score"] > best_cut["score"]:
					best_cut = cand_cuts[i]
					b = i
		if best_cut is None:
			break
		else:
			(G1, G2) = get_subgraphs(best_cut["graph"], best_cut["x"])
			best_cut["parent"].cut = best_cut["size"]
			size = size + best_cut["size"]

			if networkx.number_of_nodes(G1) == 1:
				n = Node(ind[G1.nodes()[0]])
				best_cut["parent"].add_child(n)
			elif networkx.number_of_nodes(G1) > 0:
				n = Node(None)
				
				if npol == 0:
					c = one_d_search(G1, F, k, ind)
				else:
					c = fast_search(G1, F, k, npol, ind)
				
				c["parent"] = n
				c["graph"] = G1
				cand_cuts.append(c)

				best_cut["parent"].add_child(n)
			
			if networkx.number_of_nodes(G2) == 1:
				n = Node(ind[G2.nodes()[0]])
				best_cut["parent"].add_child(n)
			elif networkx.number_of_nodes(G2) > 0:
				n = Node(None)
				
				if npol == 0:
					c = one_d_search(G2, F, k, ind)
				else:
					c = fast_search(G2, F, k, npol, ind)

				c["parent"] = n
				c["graph"] = G2
				cand_cuts.append(c)
				
				best_cut["parent"].add_child(n)
			
			del cand_cuts[b]
	
	for i in range(0, len(cand_cuts)):
		nc_recursive(cand_cuts[i]["parent"], cand_cuts[i]["graph"], ind)

	set_counts(root)
	
	return root, ind, size


