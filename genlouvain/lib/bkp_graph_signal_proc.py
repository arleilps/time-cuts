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
import statistics

def compute_eigenvectors_and_eigenvalues(L):
	lamb, U = linalg.eig(L)

	idx = lamb.argsort()   
	lamb = lamb[idx]
	U = U[:,idx]

	return U, lamb

def s(x):
	return -5 + 11*x - 6*pow(x, 2) + pow(x, 3)

def g(x):
	a = 2
	b = 2
	x_1 = 1
	x_2 = 2

	if x < x_1:
		return pow(x_1, -a)*pow(x, a)
	elif x <= x_2 and x >= x_1:
		return s(x)
	else:
		return pow(x_2, b)*pow(x, -b)

def comp_gamma():
	gn = lambda x: -1 * g(x)
	xopt = scipy.optimize.fminbound(gn, 1, 2)
	return xopt

def h(x, gamma, lamb_max, K):
	lamb_min = float(lamb_max) / K
	return gamma * math.exp(-pow(float(x/(lamb_min * 0.6)), 4))

def comp_scales(lamb_max, K, J):
	lamb_min = float(lamb_max) / K
	s_min = float(1)/lamb_max
	s_max = float(2)/lamb_min

	return numpy.exp(numpy.linspace(math.log(s_max), math.log(s_min), J))

def graph_low_pass(lamb, U, N, T, gamma, lamb_max, K):
	s = []

	for n in range(0, len(N)):
		s.append([])

	for n in range(0, len(N)):
		for m in range(0, len(U)):    
			s_n_m = 0

			for x in range(0, len(U)):
				s_n_m = s_n_m + U[n][x] * U[m][x] * h(T[-1] * lamb[x], gamma, lamb_max, K)

			s[n].append(s_n_m)

	return s

def graph_wavelets(lamb, U, N, T):
	w = []

	for t in range(0, len(T)):
		w.append([])
		for n in range(0, len(N)):
			w[t].append([])


	for t in range(0, len(T)):
		for n in range(0, len(N)):
			for m in range(0, len(U)):    
				w_t_n_m = 0

				for x in range(0, len(U)):
					w_t_n_m = w_t_n_m + U[n][x] * U[m][x] * g(T[t] * lamb[x])

				w[t][n].append(w_t_n_m)

	return w

def graph_fourier(F, U):
	lambdas = []

	for i in range(0, len(U)):
		lambdas.append(numpy.dot(F, U[:,i]))    

	lambdas = numpy.array(lambdas)

	return lambdas

def graph_fourier_inverse(GF, U):
	F = numpy.zeros(U.shape[0])
	for v in range(U.shape[0]):
		for u in range(U.shape[1]):
			F[v] = F[v] + (GF[u]*U[v][u]).real

	return F

def fourier_transform(F):
	return scipy.fftpack.fft(F)

def fourier_inverse(FT):
	return scipy.fftpack.ifft(FT)

def hammond_wavelet_transform(w, s, T, F):
	C = []

	for i in range(len(T)):
		C.append([])
		for j in range(len(F)):
			dotp = numpy.dot(F, w[i][j])
			C[i].append(dotp)

	C.append([])
	for j in range(len(F)):
		dotp = numpy.dot(F, s[j])
		C[-1].append(dotp)

	return numpy.array(C)

def hammond_wavelets_inverse(w, s, C):
	w = numpy.array(w)
	Wc = numpy.append(w, numpy.array([s]), axis=0)

	nWc = Wc[0,:,:]
	nC = C[0]
	for i in range(1,Wc.shape[0]):
		nWc = numpy.append(nWc, Wc[i,:,:], axis=0)
		nC = numpy.append(nC, C[i], axis=0)

	nWc = numpy.array(nWc)
	nC = numpy.array(nC)

	F = numpy.linalg.lstsq(nWc, nC)[0]

	return F

class Node(object):
	def __init__(self, data):
		self.data = data
		self.children = []
		self.avgs = []
		self.counts = []
		self.diffs = []
		self.scale = 0
		self.ftr = []
		self.L = []
		self.U = []

		if data is None:
			self.count = 0
		else:
			self.count = 1

	def add_child(self, obj):
		obj.scale = self.scale + 1
		self.children.append(obj)
		self.count = self.count + obj.count

def get_children(tree, part, G):
	if tree.data is not None:
		part.append(G.nodes()[tree.data])
	else:
		for c in tree.children:
			get_children(c, part, G)

def set_counts(tree):
	if tree.data is not None:
		tree.count = 1
		return 1
	else:
		count = 0
		for c in tree.children:
			count = count + set_counts(c)

		tree.count = count

		return count

def partitions_level_rec(tree, level, G, l, partitions):
	if l >= level:
		part = []
		get_children(tree, part, G)
		if len(part) > 0:
			partitions.append(part)
	else:
		if tree.data is None:
			for c in tree.children:
				partitions_level_rec(c, level, G, l+1, partitions)
		else:
			partitions.append([tree.data])
			
def partitions_level(tree, level, G):
	partitions = []
	partitions_level_rec(tree, level, G, 0, partitions)
	
	return partitions

def build_matrix(G, ind):
	M = []
	dists = networkx.all_pairs_dijkstra_path_length(G)

	M = numpy.zeros((len(G.nodes()), len(G.nodes())))

	for v1 in G.nodes():
		for v2 in G.nodes():
			M[ind[v1]][ind[v2]] = dists[v1][v2]

	return M

def select_centroids(M, radius):
	nodes = list(range(M.shape[0]))
	random.shuffle(nodes)
	nodes = nodes[:int(len(nodes)/2)]
	cents = [nodes[0]]
	mn = sys.float_info.min

	for i in range(1, len(nodes)):
		add = True
		for j in range(len(cents)):
			if M[cents[j]][nodes[i]] <= radius*mn:
				add = False
				break
		if add:
			cents.append(nodes[i])

	return cents

def coarse_matrix(M, H, cents, nodes):
	Q = numpy.zeros((len(cents), len(cents)))
	J = []
	assigns = []
	new_nodes = []

	for i in range(len(cents)):
		J.append([])
		assigns.append([])
	new_nodes.append(Node(None))

	for i in range(M.shape[0]):
		min_dist = M[i][cents[0]]
		min_cent = 0

		for j in range(1, len(cents)):
			if M[i][cents[j]] < min_dist:
				min_dist = M[i][cents[j]]
				min_cent = j

		J[min_cent].append(H[i])
		assigns[min_cent].append(i)
		new_nodes[min_cent].add_child(nodes[i])

	for i in range(len(cents)):
		if len(new_nodes[i].children) == 1:
			new_nodes[i] = new_nodes[i].children[0]

		for j in range(len(cents)):
			if i != j:
				for m in assigns[i]:
					for k in assigns[j]:
						Q[i][j] = Q[i][j] + pow(M[m][k], 2)

	Q =  normalize(Q, axis=1, norm='l1')

	return Q, J, new_nodes

def nc_recursive(A, node, node_list, ind):
	if len(node_list) < 3:
		n = Node(None)
		n.add_child(Node(ind[node_list[0]]))
		n.add_child(Node(ind[node_list[1]]))
		node.add_child(n)
	else:
		spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
		try:
			f = spectral.fit(A)
			C = spectral.fit_predict(A)
		except scipy.linalg.LinAlgError:
			 print(A.todense())
			 sys.exit()

		left = []
		right = []

		row_al = []
		col_al = []
		data_al = []
		ind_al = {}

		row_ar = []
		col_ar = []
		data_ar = []
		ind_ar = {}

		for i in range(len(C)):
			if C[i] == 1:
				ind_ar[i] = len(ind_ar)
				right.append(node_list[i])
			else:
				left.append(node_list[i])
				ind_al[i] = len(ind_al)

		for i,j,v in zip(A.nonzero()[0], A.nonzero()[1], A.data):
			if C[i] == 0 and C[j] == 0:
				row_al.append(ind_al[i])
				col_al.append(ind_al[j])
				data_al.append(v)

			if C[i] == 1 and C[j] == 1:
				row_ar.append(ind_ar[i])
				col_ar.append(ind_ar[j])
				data_ar.append(v)

		if len(left) > 1:
			AL = scipy.sparse.csr_matrix((data_al, (row_al, col_al)), shape=(len(left), len(left)))
			l = Node(None)
			nc_recursive(AL, l, left, ind)
			node.add_child(l)
		else:
			l = Node(ind[left[0]])
			node.add_child(l)

		if len(right) > 1:
			AR = scipy.sparse.csr_matrix((data_ar, (row_ar, col_ar)), shape=(len(right), len(right)))
			r = Node(None)
			nc_recursive(AR, r, right, ind)
			node.add_child(r)
		else:
			r = Node(ind[right[0]])
			node.add_child(r)

def normalized_cut_hierarchy(G):
	A = networkx.adjacency_matrix(G)
	i = 0
	ind = {}
	for v in G.nodes():
		ind[v] = i
		i = i + 1

	root = Node(None)

	nc_recursive(A, root, G.nodes(), ind)

	return root, ind

def gavish_hierarchy(G, radius):
	H = []
	nodes = []
	ind = {}
	i = 0
	for v in G.nodes():
		ind[v] = i
		nodes.append(Node(i))
		H.append(i)
		i = i + 1

	M = build_matrix(G, ind)

	while M.shape[0] > 1:
		cents = select_centroids(M, radius)
		Q, J, new_nodes = coarse_matrix(M, H, cents, nodes)
		M = Q
		H = J
		nodes = new_nodes

	return nodes[0], ind

def compute_coefficients(tree, F):
	if tree.data is None:
		avg = 0
		count = 0
		for i in range(len(tree.children)):
			compute_coefficients(tree.children[i], F)
			avg = avg + tree.children[i].avg * tree.children[i].count 
			count = count + tree.children[i].count

			if i > 0:
				tree.avgs.append(float(avg) / count)
				tree.counts.append(count)
				tree.diffs.append(2*tree.children[i].count*(tree.children[i].avg-float(avg)/count))
		tree.avgs = list(reversed(tree.avgs))
		tree.avg = float(avg) / tree.count
	else:
		tree.avg = F[tree.data]

def reconstruct_values(tree, F):
	if tree.data is None:
		avg = tree.avg * tree.count 
		count = tree.count
		for i in reversed(range(len(tree.children))):
			if i == 0:
				tree.children[i].avg = avg / tree.children[i].count
				reconstruct_values(tree.children[i], F)
			else:
				tree.children[i].avg = float(avg)/count + 0.5*float(tree.diffs[i-1]) / tree.children[i].count
				reconstruct_values(tree.children[i], F)
				count = count - tree.children[i].count
				avg = avg - tree.children[i].avg * tree.children[i].count 
				tree.avgs.append(float(avg)/count)

		tree.avgs = list(reversed(tree.avgs))
	else:
		F[tree.data] = tree.avg

def clear_tree(tree):
	tree.avg = 0 
	tree.diffs = []
	tree.avgs = []

	if tree.data is None:
		for i in range(len(tree.children)):
			clear_tree(tree.children[i])

def get_coefficients(tree, wtr):
	Q = deque()
	scales = []
	wtr.append(tree.count*tree.avg)

	Q.append(tree)

	while len(Q) > 0:
		node = Q.popleft()
		scales.append(node.scale)

		for j in range(len(node.diffs)):
			wtr.append(node.diffs[j])

		for i in range(len(node.children)):
			Q.append(node.children[i])

def set_coefficients(tree, wtr):
	Q = deque()
	tree.avg = float(wtr[0]) / tree.count
	p = 1
	Q.append(tree)

	while len(Q) > 0:
		node = Q.popleft()

		for j in range(len(node.children)-1):
			node.diffs.append(wtr[p])
			p = p + 1

		for i in range(len(node.children)):
			Q.append(node.children[i])

def gavish_wavelet_transform(tree, ind, G, F):
	wtr = []
	clear_tree(tree)
	compute_coefficients(tree, F)
	get_coefficients(tree, wtr)

	return numpy.array(wtr)

def gavish_wavelet_inverse(tree, ind, G, wtr):
	F = []

	for i in range(len(G.nodes())):
		F.append(0)

	clear_tree(tree)
	set_coefficients(tree, wtr)
	reconstruct_values(tree, F)

	return numpy.array(F)

def svd_transform(FT):
	svd_u, svd_s, svd_v = numpy.linalg.svd(FT, full_matrices=True)

	return svd_u, svd_s, svd_v

def svd_inverse(svd_u, svd_s, svd_v):
	svd_S = numpy.zeros((svd_u.shape[1], svd_v.shape[0]), dtype=complex)
	svd_S[:svd_s.shape[0], :svd_s.shape[0]] = numpy.diag(svd_s)
	FT = numpy.dot(svd_u, numpy.dot(svd_S, svd_v))

	return numpy.real(FT)

def svd_filter(svd_u, svd_s, svd_v, n):
	for k in range(n, len(svd_s)):
		svd_s[k] = 0

	return svd_s   

def pyramid_recursive(A, root, node_list, ind, k):
	if len(node_list) <= k:
		for i in range(len(node_list)):
			root.add_child(Node(ind[node_list[i]]))
		
		L = numpy.zeros((len(node_list), len(node_list)))

		for i,j,v in zip(A.nonzero()[0], A.nonzero()[1], A.data):
			L[i][j] = L[i][j] + v
			L[j][i] = L[j][i] + v

		for i in range(L.shape[0]):
			for j in range(L.shape[0]):
				if i != j:
					L[i][i] = L[i][i] + L[i][j] + 0.001
					L[i][j] = -L[i][j] - 0.001
		
		root.L = L
	else:
		spectral = SpectralClustering(n_clusters=k, affinity='precomputed')
		f = spectral.fit(A)
		C = spectral.fit_predict(A)
		new_ind = []
		new_node_list = []
		rows = []
		cols = []
		data = []
		
		for i in range(k):
			new_ind.append({})
			new_node_list.append([])
			rows.append([])
			cols.append([])
			data.append([])
		
		for i in range(len(C)):
			new_ind[C[i]][i] = len(new_ind[C[i]])
			new_node_list[C[i]].append(node_list[i])

		for i,j,v in zip(A.nonzero()[0], A.nonzero()[1], A.data):
			if C[i] == C[j]:
				rows[C[i]].append(new_ind[C[i]][i])
				cols[C[j]].append(new_ind[C[i]][j])
				data[C[i]].append(v)
		
		L = numpy.zeros((k,k))

		for i,j,v in zip(A.nonzero()[0], A.nonzero()[1], A.data):
			if C[i] != C[j]:
				L[C[i]][C[j]] = L[C[i]][C[j]] + v
				L[C[j]][C[i]] = L[C[j]][C[i]] + v

		for i in range(k):
			for j in range(k):
				if i != j:
					L[i][i] = L[i][i] + L[i][j] + 0.001
					L[i][j] = -L[i][j] - 0.001

		root.L = L
		
		for i in range(k):
			if len(new_node_list[i]) > 1:
				new_A = scipy.sparse.csr_matrix((data[i], (rows[i], cols[i])), shape=(len(new_node_list[i]), len(new_node_list[i])))
				n = Node(None)
				pyramid_recursive(new_A, n, new_node_list[i], ind, k)
				root.add_child(n)
			else:
				n = Node(ind[new_node_list[i][0]])
				root.add_child(n)

def pyramid_hierarchy(G, k):
	A = networkx.adjacency_matrix(G)
	i = 0
	ind = {}
	
	for v in G.nodes():
		ind[v] = i
		i = i + 1

	root = Node(None)
	pyramid_recursive(A, root, G.nodes(), ind, k)

	return root, ind

def compute_coefficients_pyramid(tree, F):
	if tree.data is None:
		tree.count = 0
		tree.avg = 0
		for i in range(len(tree.children)):
			compute_coefficients_pyramid(tree.children[i], F)

			tree.count = tree.count + tree.children[i].count
			tree.avg = tree.avg + tree.children[i].avg * tree.children[i].count
			
			tree.avgs.append(tree.children[i].ftr[0])
		
#		print (tree.avgs)
		tree.avgs = numpy.array(tree.avgs) 
		tree.avg = float(tree.avg) / tree.count
#		print (tree.L)
		(tree.U, lamb) = compute_eigenvectors_and_eigenvalues(tree.L)
		tree.ftr = graph_fourier(tree.avgs, tree.U)
#		print (tree.avgs)
#		print (tree.avg)
#		print (tree.U)
		tree.ftr = tree.ftr *  statistics.mean(tree.U[:,0].real)  
#		print (tree.ftr)
	else:
		tree.ftr.append(F[tree.data])
		tree.avg = F[tree.data]
		tree.count = 1

def get_coefficients_pyramid(tree, wtr):
	Q = deque()
	wtr.append(tree.ftr[0] * tree.count)
#	print(tree.ftr[0])
#	print(tree.count)
#	print(tree.ftr[0] * tree.count)
#	print("\n")


	Q.append(tree)

	while len(Q) > 0:
		node = Q.popleft()
		for j in range(1, len(node.children)):
#			print(node.ftr[j])
#			print(node.count)
#			print(node.ftr[j] * node.count)
#			print("\n")

			wtr.append(node.ftr[j] * node.count)

		for i in range(len(node.children)):
			Q.append(node.children[i])

def clear_tree_pyramid(tree):
	tree.avg = 0 
	tree.diffs = []
	tree.avgs = []
	tree.ftr = []

	if tree.data is None:
		for i in range(len(tree.children)):
			clear_tree_pyramid(tree.children[i])

def pyramid_wavelet_transform(tree, ind, G, F):
	wtr = []
	clear_tree_pyramid(tree)
	compute_coefficients_pyramid(tree, F)
	get_coefficients_pyramid(tree, wtr)

	return numpy.array(wtr)

def set_coefficients_pyramid(tree, wtr):
	Q = deque()
	tree.ftr.append(float(wtr.real[0]) / tree.count)
	p = 1
	Q.append(tree)

	while len(Q) > 0:
		node = Q.popleft()

		for j in range(len(node.children)-1):
			node.ftr.append(float(wtr.real[p]) / node.count)
			p = p + 1
		
		for i in range(len(node.children)):
			Q.append(node.children[i])

def reconstruct_values_pyramid(tree, F):
	if tree.data is None:
		tree.ftr = numpy.array(tree.ftr) / statistics.mean(tree.U[:,0].real)
		tree.avgs = graph_fourier_inverse(tree.ftr, tree.U) 
#		print (tree.avgs)
#		print (tree.avgs)
#		print (tree.U)
#		print (tree.ftr)
		
		for i in range(len(tree.children)):
			tree.children[i].avg = float(tree.avgs[i]) / tree.children[i].count
			tree.children[i].ftr.insert(0, tree.avgs[i])
			reconstruct_values_pyramid(tree.children[i], F)
	else:
		F[tree.data] = tree.avg

def pyramid_wavelet_inverse(tree, ind, G, wtr):
	F = numpy.zeros(len(G.nodes()))

	clear_tree_pyramid(tree)
	set_coefficients_pyramid(tree, wtr)
	reconstruct_values_pyramid(tree, F)

	return numpy.array(F)

def set_weight_graph(G, F=None):
	ind = {}
	i = 0

	for v in G.nodes():
		ind[v] = i
		i = i + 1

	for e in G.edges():
		v1 = e[0]
		v2 = e[1]

		if F is None:
			G[v1][v2]['weight'] = 1.0
		else:
			G[v1][v2]['weight'] = math.exp(-abs(F[ind[v1]]-F[ind[v2]]))
			if G[v1][v2]['weight'] < 0.0001:
				G[v1][v2]['weight'] = 0.
			
#			print (G[v1][v2]['weight'])

