import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import sparse

from lib.vis import *
from lib.time_graph import *
from scipy import linalg
from numpy.linalg import eigh

def L2_error(F, F_approx):
	e = 0
	
	for i in range(F.shape[0]):
		e = e + ((F[i]-F_approx[i])**2).sum()

	return float(e)

def graph_fourier_transform(G, F, k):
	G.index_vertex()
	lambdas = []
	L = create_laplacian_matrix(G)
	
	try:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(L, k=k, which='SA')
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigs(L, k=k, which='SM')
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			L = L.todense()
			(eigvals, eigvecs) = scipy.linalg.eigh(L, eigvals=(0, k))

	for i in range(0, k):
		lambdas.append(numpy.dot(F, eigvecs[:,i]))
	
	F_approx = numpy.zeros(G.number_of_nodes())

	for i in range(0, k):
		for v in range(G.number_of_nodes()):
			F_approx[v] = F_approx[v] + (lambdas[i] * eigvecs[v,i]).real
	
	return F_approx

def create_signal_matrix(G, F):
	row = []
	column = []
	value = []

	for t in range(G.num_snaps()-1):
		for k in range(0, 2):
			for v in range(G.size()):
				for u in range(G.size()):
					(vi, ti) = G.rev_index_pos(G.size()*t + v)
					(vj, tj) = G.rev_index_pos(G.size()*(t+k) + u)
						
					if G.active(vi, t) and G.active(vj, t+k):
						row.append(G.size()*t + v)
						column.append(G.size()*(t+k) + u)
						value.append(math.pow(F[G.size()*t + v]-F[G.size()*(t+k) + u], 2))
					
						column.append(G.size()*t + v)
						row.append(G.size()*(t+k) + u)
						value.append(math.pow(F[G.size()*t + v]-F[G.size()*(t+k) + u], 2))

	sz = G.num_snaps() * G.size()
	S = scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

	return S

def sweep_signal(G, x, F, alpha):
	best_score = -sys.float_info.max
	num_swaps = 0.
	num_edges = 0.
	sorted_x = numpy.argsort(x)
	edges_cut = 0
	swaps = 0
	nodes_one = []
	sizes_one = []
	sums_one = []
	sums_total = []
	den = 0
	best_edges_cut = G.number_of_edges()
	best_swaps = G.size() * G.num_snaps()
	best_num_edges = G.number_of_edges()
	best_num_swaps = G.size() * G.num_snaps()

	for t in range(G.num_snaps()):
		nodes_one.append({})
		sizes_one.append(0)
		sums_one.append(0)
		s = 0
		for v in range(G.size()):
			(vi,ti) = G.rev_index_pos(G.size()*t + v)
			if G.active(vi, t):
				s = s + F[G.size()*t + v]
		
		sums_total.append(s)

	for i in range(x.shape[0]-1):
		(v,t) = G.rev_index_pos(sorted_x[i])
		if G.active(v, t):
			den = den - sizes_one[t] * (G.number_of_nodes(t) - sizes_one[t])
			sizes_one[t] = sizes_one[t] + 1
			sums_one[t] = sums_one[t] + F[sorted_x[i]]
			den = den + sizes_one[t] * (G.number_of_nodes(t) - sizes_one[t])

			nodes_one[t][v] = True
			
			for u in G.graphs[t].neighbors(v):
				if G.active(u, t):
					if u not in nodes_one[t]:
						edges_cut = edges_cut + G.graphs[t][v][u]["weight"]
						num_edges = num_edges + 1.
					else:
						edges_cut = edges_cut - G.graphs[t][v][u]["weight"]
						num_edges = num_edges - 1.

			if t+1 < G.num_snaps():
				if G.active(v, t+1):
					if v not in nodes_one[t+1]:
						swaps = swaps + G.swap_cost_vertex(v, t)
						num_swaps = num_swaps + 1.
					else:
						swaps = swaps - G.swap_cost_vertex(v, t)
						num_swaps = num_swaps - 1.

			if t > 0:
				if G.active(v, t-1):
					if v not in nodes_one[t-1]:
						swaps = swaps + G.swap_cost_vertex(v, t-1)
						num_swaps = num_swaps + 1.
					else:
						swaps = swaps - G.swap_cost_vertex(v, t-1)
						num_swaps = num_swaps - 1.

			den_reg = den + alpha * (swaps + edges_cut)
	
			num = 0.
			for t in range(G.num_snaps()):
				at = sizes_one[t] * (sums_total[t]-sums_one[t]) - (G.number_of_nodes(t)-sizes_one[t]) * sums_one[t]
				num = num + math.pow(at, 2) 
				
				for k in range(G.num_snaps()):
					if k != t:
						ak = sizes_one[k] * (sums_total[k]-sums_one[k]) - (G.number_of_nodes(t)-sizes_one[k]) * sums_one[k]
						num = num + at * ak

			if den_reg > 0:
				score = num / den_reg  
			else:
				score = -sys.float_info.max


			if score >= best_score:
				best_score = score
				best = i
				best_edges_cut = edges_cut
				best_swaps = swaps
				best_num_edges = num_edges
				best_num_swaps = num_swaps

	vec = numpy.zeros(G.size() * G.num_snaps())

	for i in range(x.shape[0]):
		(v,t) = G.rev_index_pos(sorted_x[i])
		
		if G.active(v,t):
			if i <= best:
				vec[sorted_x[i]] = -1.
			else:
				vec[sorted_x[i]] = 1.
		else:
			vec[sorted_x[i]] = 0.

	return {"cut": vec, "score": best_score, "edges": best_edges_cut, "swaps": best_swaps, "num_edges": best_num_edges, "num_swaps": best_num_swaps}

def wavelet_cut(G, F, alpha=1.):
	L = create_laplacian_matrix(G)
	C = create_c_matrix(G)
	S = create_signal_matrix(G, F)
	
	CSC = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(C, S), C)
	isqrtCL = sqrtmi( (C+alpha * L).todense())
	M = numpy.dot(numpy.dot(isqrtCL, CSC.todense()), isqrtCL)
	
	try:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(M, k=1)
		vec = eigvecs[:,0]
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1)
			vec = eigvecs[:,0]
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			M = M.todense()
			(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(0,0))
			vec = eigvecs[:,0]

	return sweep_signal(G, vec, F, alpha)

def multi_cut_signal(G, K, F, alpha=1.):
	G.index_vertex()
	root = Node(None)
	k = 1
	num_edges = 0
	num_swaps = 0
	cand_cuts = []
	cut = wavelet_cut(G, F, alpha)
	cut["parent"] = root
	cut["graph"] = G

	cand_cuts.append(cut)
	
	while k < K and len(cand_cuts) > 0:
		best_cut = None
		b = 0
		for c in range(len(cand_cuts)):
			if best_cut is None or cand_cuts[c]["score"] > best_cut["score"]:
				best_cut = cand_cuts[c]
				b = c
		
		(G1,G2) = best_cut["graph"].break_graph_cut(best_cut["cut"])
		num_edges = num_edges + best_cut["num_edges"]
		num_swaps = num_swaps + best_cut["num_swaps"]

		
		if G1.number_of_nodes() == 1:
			(v, t) = G1.active_nodes()[0]
			i = G.index_vertex(t, v)
			n = Node([i])
			best_cut["parent"].add_child(n)
		else:
			n = Node(None)
			cut = wavelet_cut(G1, F, alpha)
			cut["parent"] = n
			cut["graph"] = G1
			
			cand_cuts.append(cut)
			best_cut["parent"].add_child(n)
		
		if G2.number_of_nodes() == 1:
			(v, t) = G2.active_nodes()[0]
			i = G.index_vertex(t, v)
			n = Node([i])
			best_cut["parent"].add_child(n)
		else:
			n = Node(None)
			cut = wavelet_cut(G2, F, alpha)
			cut["parent"] = n
			cut["graph"] = G2
			
			cand_cuts.append(cut)
			best_cut["parent"].add_child(n)

		del cand_cuts[b]
		
		k = k + 1

	for c in cand_cuts:
		parent = c["parent"]
		data = []
		for (v,t) in c["graph"].active_nodes():
			i = G.index_vertex(t, v)
			data.append(i)

		parent.data = data

	c = {}
	c["part"] = get_partitions_tree(root, G)
	c["assign"] = get_partition_assign(G, c["part"])
	c["num_edges"] = num_edges
	c["num_swaps"] = num_swaps

	return c

def temporal_graph_transform(G, F, K, alpha=1.):
	G.index_vertex()
	c = multi_cut_signal(G, K, F, alpha)
	size = K + int(math.ceil(float(c["num_edges"] * math.log2(G.number_of_edges())) / 64 + float(c["num_swaps"] * math.log2(G.size() * (G.num_snaps()-1))) / 64))
	c["size"] = size
	avgs = []

	for p in range(len(c["part"])):
		s = 0.
		for i in c["part"][p]:
			s = s + F[i]
		avgs.append(s/len(c["part"][p]))
	
	tr = numpy.zeros(F.shape[0])

	for i in range(F.shape[0]):
		tr[i] = avgs[int(c["assign"][i])]

	c["transform"] = tr

	return c

