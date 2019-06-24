import networkx
import math
import numpy
import scipy
import sys
from scipy import linalg

from numpy import log, mean, dot, diag, sqrt
from numpy.linalg import eigh
from lib.graph_signal_proc import *

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


def sweep_vec(x, A, C, L, beta):
	best_val = 0.
	sorted_x = numpy.argsort(x)
	
	for i in range(x.shape[0]):
		cand_x = []
		pivot = x[sorted_x[i]]
		
		for j in range(x.shape[0]):
			if x[j] < pivot:
				cand_x.append(-1.)
			else:
				cand_x.append(1.)
		
		cand_x = numpy.array(cand_x)
		
		den = (numpy.dot(numpy.dot(cand_x, C), cand_x) + beta * numpy.dot(numpy.dot(cand_x, L), cand_x))
		
		if den > 0:
			val = -numpy.dot(numpy.dot(numpy.dot(numpy.dot(cand_x, C), A), C), cand_x) / den
		else:
			val = 0
		
		if val >= best_val:
			best_cand = cand_x
			best_val = val
		
	return best_cand, best_val

def laplacian_complete(n):
	C = numpy.ones((n, n))
	C = -1 * C
	D = numpy.diag(numpy.ones(n))
	C = (n)*D + C
	
	return C

def weighted_adjacency_complete(F):
	A = []
	for i in range(F.shape[0]):
		A.append([])
		for j in range(F.shape[0]):
			A[i].append(pow(F[i]-F[j],2))
			
	return numpy.array(A)
	
def spectral_cut(L, F, beta):
	C = laplacian_complete(F.shape[0])
	A = weighted_adjacency_complete(F)
	
	isqrtCL = sqrtmi(C + (beta * L))
	
	M = numpy.dot(numpy.dot(isqrtCL, numpy.dot(numpy.dot(C,A), C)), isqrtCL)
	(eigvals, eigvecs) = scipy.linalg.eigh(M,eigvals=(0,0))
	
	x = numpy.asarray(numpy.dot(eigvecs[:,0], isqrtCL))[0,:]
	(x, score) = sweep_vec(x, A, C, L, beta)
	
	res = {}
	res["x"] = x
	res["size"] = numpy.dot(numpy.dot(x, L), x)[0,0] / 4
	res["score"] = score
	
	#print(res["x"])
	#print(res["size"])
	#print(L)

	return res

def get_new_functions(P1, P2, ind, F):
	F1 = []
	F2 = []
	for i in range(len(P1)):
		F1.append(F[ind[P1[i]]])
	
	for i in range(len(P2)):
		F2.append(F[ind[P2[i]]])

	return numpy.array(F1), numpy.array(F2)

def optimal_wavelet_basis(G, F, beta, k):
	ind = {}
	node_list = []
	i = 0
	
	for v in G.nodes():
		ind[v] = i
		i = i + 1

	L = networkx.laplacian_matrix(G).todense()
	root = Node(None)
	size = 0
	cand_cuts = []
	c = spectral_cut(L, F, beta)
	c["parent"] = root
	c["list"] = G.nodes()
	
	cand_cuts.append(c)
	
	while size <= k and len(cand_cuts) > 0:
		best_cut = None
		b = 0
		
		for i in range(0, len(cand_cuts)):
			if cand_cuts[i]["size"] + size <= k:
				if best_cut is None or cand_cuts[i]["score"] > best_cut["score"]:
					best_cut = cand_cuts[i]
					b = i
		if best_cut is None:
			break
		else:
			(P1, P2) = get_partitions(best_cut["x"], best_cut["list"])
			(LL, LR) = get_new_laplacians(L, P1, P2, ind)
			(FL, FR) = get_new_functions(P1, P2, ind, F)
			best_cut["parent"].cut = best_cut["size"]
			size = size + best_cut["size"]

			if len(P1) == 1:
				n = Node(ind[P1[0]])
				best_cut["parent"].add_child(n)
			elif len(P1) > 0:
				n = Node(None)
				c = spectral_cut(LL, FL, beta)
				c["parent"] = n
				c["list"] = P1
				cand_cuts.append(c)

				best_cut["parent"].add_child(n)
			
			if len(P2) == 1:
				n = Node(ind[P2[0]])
				best_cut["parent"].add_child(n)
			elif len(P2) > 0:
				n = Node(None)
				c = spectral_cut(LR, FR, beta)
				c["parent"] = n
				c["list"] = P2
				cand_cuts.append(c)
				
				best_cut["parent"].add_child(n)
			
			del cand_cuts[b]
	
	for i in range(0, len(cand_cuts)):
		(LL, LR) = get_new_laplacians(L, cand_cuts[i]["list"], [], ind)
		nc_recursive(L, LL, cand_cuts[i]["parent"], cand_cuts[i]["list"], ind)

	set_counts(root)
	
	return root, ind, size

def L2_error(F, F_approx):
	e = 0
	for i in range(F.shape[0]):
		e = e + ((F[i]-F_approx[i])**2).sum()
 
	return float(e)

def best_tree_avg(G, F, k, beta, n):
	trees = []
	min_error = sys.float_info.max
	min_tree = None
	min_ind = None
	
	for a in range(F.shape[0]):
		(tree, ind, s) =  optimal_wavelet_basis(G, F[a], beta, k)
		error = 0

		for i in range(F.shape[0]):
			wtr = gavish_wavelet_transform(tree, ind, G, F[i])
			coeffs = {}
			
			for j in range(len(wtr)):
				coeffs[j] = abs(wtr[j])
				sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)
			
			wtr_copy = numpy.copy(wtr)
			
			for k in range(n, len(sorted_coeffs)):
				j = sorted_coeffs[k][0]
			
				wtr_copy[j] = 0.0
			 
			F_appx = gavish_wavelet_inverse(tree, ind, G, wtr_copy)
			error = error + L2_error(F[i], F_appx)

		
		if error < min_error:
			min_tree = tree
			min_ind = ind
			min_error = error

	return min_tree, min_ind



