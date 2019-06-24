import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import sparse

from lib.vis import *
from lib.graph_signal_proc import *
from lib.optimal_cut import *
from scipy import linalg
import matplotlib.pyplot as plt
from IPython.display import Image
from numpy.linalg import eigh
from heapq import heappush, heappop

class TimeGraph(object):
	"""
		Class for managing temporal/dynamic graphs
	"""
	def __init__(self, swp_cost=1.):
		"""
			Initialization.
			Input:
				* swp_cost: cost of swaping a node across cuts
				* t_max: maxmum number of snapshots (not used)
		"""
		self.graphs = [networkx.Graph()]
		self.swp_cost = swp_cost
		self.idx_vertex = []
		self._node_list = []
	
	def nodes(self):
		return self._node_list

	def number_of_nodes(self):
		return self.size() * self.num_snaps()

	def number_of_edges(self):
		n = 0
		for t in range(len(self.graphs)):
			n = n + self.graphs[t].number_of_edges()

		return n
	
	def index_vertex(self, t=None, v=None):
		"""
			Index for vertices. Returns a unique integer for a pair <t,v>.
			Vertices are indexed based on first snapshot G.nodes() list (from networkx) one snapshot after another.
				Input:
					* t: timestamp of the snapshot
					* v: vertex name (from networkx)
		"""
		if len(self.idx_vertex) != self.num_snaps():
			i = 0
			self.idx_vertex = []
			self.node_list = []
				
			for u in self.graphs[0].nodes():
				self._node_list.append(u)

			for j in range(len(self.graphs)):
				self.idx_vertex.append({})
				
				for u in self.graphs[0].nodes():
					self.idx_vertex[j][u] = i
					i = i + 1
		if t is not None and v is not None:	
			return self.idx_vertex[t][v]

	def swap_cost(self):
		"""
			Returns the swap cost (constant)
		"""
		return self.swp_cost

	def extend(self, t):
		"""
			Extends the temporal graph up to t timestamps
			Input:
				* t: timestamp for extension
		"""
		while len(self.graphs) <= t:
			self.graphs.append(networkx.Graph())

			for v in self.graphs[0].nodes():
				self.graphs[-1].add_node(v)
		            
	def add_edge(self, v1,v2,t,w=1.):
		"""
			Adds edge (v1,v2) to temporal graph. Graph is undirected
			Input:
				* v1: vertex v1
				* v2: vertex v2
				* t: timestamp
				* w: edge weight
		"""
		self.extend(t)
		
		if v1 not in self.graphs[0]:
			self.add_node(v1)
						                
		if v2 not in self.graphs[0]:
			self.add_node(v2)
										           
		self.graphs[t].add_edge(v1, v2, weight=w)

	def add_node(self, v):
		"""
			Adds node to temporal graph.
			Input:
				* v: node
		"""
		for t in range(len(self.graphs)):
			self.graphs[t].add_node(v)

	def remove_node(self, v):
		"""
		"""
		for t in range(len(self.graphs)):
			self.graphs[t].remove_node(v)

	def remove_snapshot(self, t):
		"""
		"""
		del(self.graphs[t])

	def degree(self, v):
		"""
			Computes degree of v in the temporal graph,
			which is the sum of the degrees over snapshots
		"""
		d = 0
		for t in range(len(self.graphs)):
			d = d + self.graphs[t].degree(v, weight="weight")
		
		return d

	def add_snap(self, G, t):
		"""
			Adds snapshot G at timestamp t. If t already exists, edges are added.
			Input:
				* G: Snapshot
				* t: timestamp
		"""
		for v in G.nodes():
			self.add_node(v)

		for (v1,v2) in G.edges():
			self.add_edge(v1,v2,t)

	def window(self, tb, te):
		"""
			Returns a window [tb,te] of a temporal graph.
			Input:
				* tb: begin
				* te: end
			Ouput:
				* G: G[tb,te]
		"""
		G = TimeGraph()
		for t in range(tb, te+1):
			G.add_snap(self.snap(t), t-tb)

		return G
	
	def size(self):
		"""
			Number of nodes of the graph.
		"""
		return networkx.number_of_nodes(self.graphs[0])
	     
	def num_snaps(self):
		"""
			Number of snapshots in the graph
		"""
		return len(self.graphs)

	def snap(self,t):
		"""
			Temporal graph snapshot G[t]
		"""
		return self.graphs[t]

	def rev_index_pos(self, p):
		"""
			Returns vertex with given unique ID and its respective timestamp. 
			Used to index a large vector with a position for each vertex in the temporal graph.
			Vertices are indexed based on first snapshot G.nodes() list (from networkx) one snapshot after another.
			Input:
				* p: vertex id
			Output:
				* vertex name, from networkx Graph
				* timestamp
		"""
		return self._node_list[p % self.size()], int(p / self.size())

	def rev_index_pos_single(self, p):
		"""
			Maps given vertex vertex id to a single vector. The same vertex in different snapshots should all map to the same position.
			Input:
				* p: vertex id
			Output:
				* vertex id
				* timestamp
		"""
		return p % self.size(), int(p / self.size())

	def set_values(self, f):
		"""
			Sets vallues in a vector f to vertices in the temporal graph. Position are consistent with the index.
			Input:
				* f: values
		"""
		i = 0
		for t in range(len(self.graphs)):
			for v in self.graphs[0].nodes():
				self.graphs[t].node[v]["value"] = f[self.index_vertex(t, v)]
				i = i + 1

	def separate_values(self, f):
		"""
			Separates values in f for each snasphot.
			Input:
				* f: values
			Output:
				* #snapshots x #vertices matrix with values
		"""
		c = numpy.zeros((self.num_snaps(), self.graphs[0].number_of_nodes()))
		
		for i in range(f.shape[0]):
			p,t = self.rev_index_pos_single(i)
			c[t][p] = f[i]
		
		return c

	def aggregate_values(self, f):
		"""
			Aggregates values for different cuts into a single temporal cut.
			Temporal graph indexing assumed.
		"""
		c = numpy.zeros(self.num_snaps() * self.size())

		for i in range(c.shape[0]):
			p, t = self.rev_index_pos_single(i)
			c[i] = f[t][p]

		return c

	def name_stacked(self, v, t):
		return str(t) + "--" + str(v)
	
	def parse_name_stacked(self, name):
		vec = name.split("--")

		return vec[1], int(vec[0])
	
	def build_stacked_graph(self):
		SG = networkx.Graph()
		
		for t in range(len(self.graphs)):
			for e in self.graphs[t].edges():
				v1 = str(t) + "--" + str(e[0])
				v2 = str(t) + "--" + str(e[1])
				SG.add_edge(v1, v2, weight=self.graphs[t][e[0]][e[1]]["weight"])
		
		for t in range(len(self.graphs)-1):
			for v in self.graphs[0].nodes():
				v1 = str(t) + "--" + str(v)
				v2 = str(t+1) + "--" + str(v)
				
				SG.add_edge(v1, v2, weight=self.swp_cost)

		return SG

	def write(self, out_file_name):
		out_file = open(out_file_name, 'w')
		
		for t in range(len(self.graphs)):
			for e in self.graphs[t].edges():
				out_file.write(str(e[0])+","+str(e[1])+","+str(t)+","+str(self.graphs[t][e[0]][e[1]]['weight'])+"\n")	

		out_file.close()

def sweep_single_qp(G, x, single_side=False):
	"""
		Sweep algorithm for QP formulation and single snapshot. Graph is extended with supernodes representing cuts
		on previous and/or next snapshot.
		Input:
			* G: graph
			* x: vector
			* single_side: true if either last or first snapshot
		Ouput:
			* Rounded vector
	"""
	best_val = sys.float_info.max
	sorted_x = numpy.argsort(x) #positions sorted by increasing value of x
	size_one = 0
	edges_cut = 0
	nodes_one = {}
	best_edges_cut = 0
	best_size_one = 0

	if single_side:
		#One supernode (out of two) is assigned to one side of the cut.
		start = 1
		end = G.number_of_nodes()-1
		nodes_one[G.nodes()[sorted_x[0]]] = True
		edges_cut = G.degree(G.nodes()[sorted_x[0]], weight="weight")
	else:
		#Two supernodes (out of four) are assigned to one side of the cut.
		start = 2
		end = G.number_of_nodes()-2
		nodes_one[G.nodes()[sorted_x[0]]] = True
		nodes_one[G.nodes()[sorted_x[1]]] = True
		edges_cut = G.degree(G.nodes()[sorted_x[0]], weight="weight") + G.degree(G.nodes()[sorted_x[1]], weight="weight")

	for i in range(start, end):
		size_one = size_one + 1
		
		nodes_one[G.nodes()[sorted_x[i]]] = True
		u = G.nodes()[sorted_x[i]]
		
		for v in G.neighbors(u):
			if v not in nodes_one:
				edges_cut = edges_cut + G[v][u]["weight"]
			else:
				edges_cut = edges_cut - G[v][u]["weight"]

		den = size_one * (end-start-size_one)

		if den > 0:
			val = float(edges_cut) / den
		else:
			val = networkx.number_of_nodes(G)
		
		if val <= best_val:
			best_cand = i
			best_val = val
			best_edges_cut = edges_cut
			best_size_one = size_one

	vec = []

	vec = numpy.zeros(networkx.number_of_nodes(G))

	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return {"cut": vec, "edges": best_edges_cut, "size_one": best_size_one}

def sweep_single(G, x, nodes_list=None):
	"""
		Eigenvector sweep rounding on a single graph/snapshot.
		Input:
			* G: graph
			* x: vector
			* nodes_list: list of nodes (order), G.nodes() is applied if None
		Output:
			* rounded vector
	"""
	best_val = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	nodes_one = {}
	best_edges_cut = 0
	best_size_one = 0

	if nodes_list is None:
		nodes_list = G.nodes()

	for i in range(x.shape[0]):
		size_one = size_one + 1
		
		nodes_one[nodes_list[sorted_x[i]]] = True
		u = nodes_list[sorted_x[i]]
		
		for v in G.neighbors(u):
			if v not in nodes_one:
				edges_cut = edges_cut + G[v][u]["weight"]
			else:
				edges_cut = edges_cut - G[v][u]["weight"]
			
		den = size_one * (networkx.number_of_nodes(G)-size_one)

		if den > 0:
			val = float(edges_cut) / den
		else:
			val = networkx.number_of_nodes(G)

		if val <= best_val:
			best_cand = i
			best_val = val
			best_edges_cut = edges_cut
			best_size_one = size_one

	vec = []

	vec = numpy.zeros(networkx.number_of_nodes(G))

	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return {"cut": vec, "edges": best_edges_cut, "size_one": best_size_one}

def sweep(G, x):
	best_score = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	swaps = 0
	nodes_one = []
	sizes_one = []
	den = 0

	for t in range(G.num_snaps()):
		nodes_one.append({})
		sizes_one.append(0)

	for i in range(x.shape[0]):
		(v,t) = G.rev_index_pos(sorted_x[i])
		den = den - sizes_one[t] * (G.size() - sizes_one[t])
		sizes_one[t] = sizes_one[t] + 1
		den = den + sizes_one[t] * (G.size() - sizes_one[t])

		nodes_one[t][v] = True
		
		for u in G.graphs[t].neighbors(v):
			if u not in nodes_one[t]:
				edges_cut = edges_cut + G.graphs[t].edge[v][u]["weight"]
			else:
				edges_cut = edges_cut - G.graphs[t].edge[v][u]["weight"]

		if t+1 < G.num_snaps():
			if v not in nodes_one[t+1]:
				swaps = swaps + G.swap_cost()
			else:
				swaps = swaps - G.swap_cost()

		if t > 0:
			if v not in nodes_one[t-1]:
				swaps = swaps + G.swap_cost()
			else:
				swaps = swaps - G.swap_cost()
		
		if den > 0:
			score = float(edges_cut + swaps) / den
		else:
			score = sys.float_info.max

		if score <= best_score:
			best_score = score
			best = i
			best_edges_cut = edges_cut
			best_swaps = swaps

	vec = numpy.zeros(G.size() * G.num_snaps())

	for i in range(x.shape[0]):
		if i <= best:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return {"cut": vec, "score": best_score, "edges": best_edges_cut, "swaps": best_swaps}

def evaluate_cut(G, x):
	edges_cut = 0
	swaps = 0
	den = 0

	for t in range(G.num_snaps()):
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]
			
			cut = x[G.index_vertex(t, v1)] + x[G.index_vertex(t, v2)]

			if abs(cut) < 1.:
				edges_cut = edges_cut + G.snap(t)[v1][v2]["weight"]
			
	for t in range(G.num_snaps()):
		size_one = 0
		for v in G.snap(t).nodes():
			if t < G.num_snaps()-1:
				cut = x[G.index_vertex(t, v)] + x[G.index_vertex(t+1, v)]
			
				if abs(cut) < 1.:
					swaps = swaps + G.swap_cost()
			
			if x[G.index_vertex(t, v)] < 0:
				size_one = size_one + 1

		den = den + size_one * (G.size() - size_one)
	
	return edges_cut, swaps, den

def aggreg_opt_cuts(G, c):
	"""
		Aggregates independent cuts into a temporal graph cut using shortest paths.
		Input:
			* G: Temporal graph
			* c: Isolated cuts
		Output:
			* Temporal graph cut
	"""
	edges_cut = 0

	#Computes total number of edges cut in the graph
	for i in range(len(c)):
		edges_cut = edges_cut + c[i]["edges"]

	#Builds a graph with edge costs represented by cut swaps.
	Gs = networkx.Graph()
	Gs.add_edge(0, 1, weight=0.)
	Gs.add_edge(0, 2, weight=0.)

	for i in range(1, len(c)):
		w = float(numpy.absolute(c[i-1]["cut"] - c[i]["cut"]).sum()) / 2

		Gs.add_edge(2*i-1, 2*i+1, weight=w)
		Gs.add_edge(2*i, 2*i+2, weight=w)
		
		Gs.add_edge(2*i-1, 2*i+2, weight=(G.size()-w))
		Gs.add_edge(2*i, 2*i+1, weight=(G.size()-w))

	Gs.add_edge(2*len(c)-1, 2*len(c)+1, weight=0.)
	Gs.add_edge(2*len(c), 2*len(c)+1, weight=0.)
	
	#Computes shortest paths in the swap graph
	p = networkx.shortest_path(Gs, source=0, target=2*len(c)+1, weight="weight")
	vec = numpy.array([])
	swaps = 0
	den = 0

	for i in range(1, len(p)-1):
		if p[i] % 2 == 0:
			c[i-1]["cut"] =  -1 * c[i-1]["cut"]
		
		s1 = 0

		for j in range(c[i-1]["cut"].shape[0]):
			if c[i-1]["cut"][j] < 0:
				s1 = s1 + 1
		
		den = den + s1 * (G.size() - s1)

		swaps = swaps + Gs.edge[p[i-1]][p[i]]["weight"]
		vec = numpy.concatenate((vec, c[i-1]["cut"]), axis=0)

	score = float(edges_cut + swaps) / den
	
	return {"cut": vec, "score": score, "edges": edges_cut, "swaps": swaps}



def create_laplacian_matrix(G):
	"""
		Creates laplacian matrix for temporal graph.
		This is the laplacian of the layered/stacked graph with edges
		connecting the same node within consecutive snapshots
	"""
	row = []
	column = []
	value = []

	for t in range(G.num_snaps()):
		Ag = networkx.adjacency_matrix(G.snap(t), G.nodes(), weight='weight')
		for (i,j) in zip(*scipy.nonzero(Ag)):
			row.append(G.size()*t + i)
			column.append(G.size()*t + j)
			
			if i != j:
				value.append(Ag[i,j])
				
	
	for t in range(G.num_snaps()-1):
		for v in range(G.size()):
			row.append(t*G.size() + v)
			column.append((t+1)*G.size() + v)
			value.append(G.swap_cost())
			
			column.append(t*G.size() + v)
			row.append((t+1)*G.size() + v)
			value.append(G.swap_cost())

	sz = G.num_snaps() * G.size()
	A =  scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

	D = A.sum(axis=0)
	
	row = []
	column = []
	value = []

	for i in range(D.shape[1]):
		row.append(i)
		column.append(i)
		value.append(D[0,i])

	D =  scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

	return D-A

def create_modularity_matrix(G, omega):
	"""
		Creates modularity matrix for temporal graph.
	"""
	G.index_vertex()
	row = []
	column = []
	value = []

	for t in range(G.num_snaps()):
		Ag = networkx.adjacency_matrix(G.snap(t), G.nodes(), weight='weight')
		m = 2 * scipy.sparse.coo_matrix.sum(Ag)
		K = numpy.zeros(Ag.shape[0])

		for i in range(Ag.shape[0]):
			K[i] = scipy.sparse.coo_matrix.sum(Ag[i])

		for i in range(Ag.shape[0]):
			for j in range(Ag.shape[0]):
				if i != j:
					row.append(G.size()*t + i)
					column.append(G.size()*t + j)
					v = Ag[i,j] - K[i]*K[j]/(2*m)	
					value.append(v)
				
	for t in range(G.num_snaps()-1):
		for v in range(G.size()):
			row.append(t*G.size() + v)
			column.append((t+1)*G.size() + v)
			value.append(omega)
			
			column.append(t*G.size() + v)
			row.append((t+1)*G.size() + v)
			value.append(omega)

	sz = G.num_snaps() * G.size()
	M =  scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

	return M

def merge_communities_cut(G, P, omega):
	comms = []
	num_comms = 0
	
	for v in P:
		for t in P[v]:
			if P[v][t]+1 >= num_comms:
				num_comms = P[v][t]+1
	for i in range(num_comms):
		comms.append([])
	
	for v in P:
		for t in P[v]:
			comms[P[v][t]].append(G.index_vertex(t,v))

	B = create_modularity_matrix(G, omega)	

	while num_comms > 2:
		best_pair=None
		best_score = -sys.float_info.max
		for c1 in range(len(comms)):
			for c2 in range(len(comms)):
				if c1 < c2:
					score = 0
					for v in comms[c1]:
						for u in comms[c2]:
							if u != v:
								score = score + B[u,v]

					if score > best_score:
						best_score = score
						best_pair = (c1,c2)

		c1 = best_pair[0]
		c2 = best_pair[1]
		C = comms[c1] + comms[c2]
		if c1 > c2:
			del comms[c1]
			del comms[c2]
		else:
			del comms[c2]
			del comms[c1]
	
		comms.append(C)

		num_comms = num_comms - 1
	vec = numpy.ones(G.number_of_nodes())
	size_one = 0
	for v in comms[0]:
		vec[v] = -1.
		size_one = size_one + 1

	(edges_cut, swaps, den) = evaluate_cut(G, vec)
	score = float(edges_cut + swaps) / den
	
	return {"cut": vec, "score": score, "edges": edges_cut, "swaps": swaps}

def create_constraint_matrix(n, vb1, vb2, va1=0, va2=0):
	"""
		Creating constraint matrix C. values should add to 0. and 
		in particular super-node values should add to 0 to distinguish
		two sides of the prefix and suffix cut. Order does not matter.
		Input:
			* n: size of the graph
			* vb1: index of super node one of previous cut
			* vb2: index of super node two of previous cut
			* va1: index of super node one of next cut (optional)
			* va2: index of super node two of next cut (optional)
		Output:
			* Constraint matrix
	"""
	row = []
	column = []
	value = []

	for i in range(n):
		row.append(0)
		column.append(i)
		value.append(1.)

	row.append(1)
	column.append(vb1)
	value.append(1.)

	row.append(1)
	column.append(vb2)
	value.append(1.)
	
	if va1 != 0 and va2 != 0:
		row.append(2)
		column.append(va1)
		value.append(1.)

		row.append(2)
		column.append(va2)
		value.append(1.)
	
		return scipy.sparse.csr_matrix((value, (row, column)), shape=(3, n), dtype=float)
	else:
		return scipy.sparse.csr_matrix((value, (row, column)), shape=(2, n), dtype=float)

def create_c_matrix(G):
	"""
		Creates Laplacian matrix of complete graphs for each layer of the temporal graph.
		Input:
			* Graph: Temporal graph
		Output:
			* Laplacian matrix (sparse)
	"""
	row = []
	column = []
	value = []
	
	for t in range(G.num_snaps()):
		for i in range(G.size()):
			for j in range(G.size()):
				row.append(t*G.size() + i)
				column.append(t*G.size() + j)
				
				if i == j:
					value.append(G.size()-1)
				else:
					value.append(-1.)
	
	sz = G.num_snaps() * G.size()
	
	return scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

def temporal_cut_prod(G):
	"""
		Computes temporal cut via eigenvectors of CLC^T matrix, where L is the Laplacian
		of the temporal graph and C is the Laplacian of layers that are complete graphs.
		Input: 
			* G: Temporal graph
		Output:
			* temporal cut
	"""
	G.index_vertex()
	L = create_laplacian_matrix(G)
	C = create_c_matrix(G)
	
	M = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(C, L), C)
	
	try:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=G.num_snaps()+1, which='SM')
		idx = eigvals.argsort()[::-1]
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:,idx]
		x = eigvecs[:,0]
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		Md = M.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Md, eigvals=(G.num_snaps(), G.num_snaps()))
		x = numpy.asarray(scipy.dot(eigvecs[:,0], C.todense()))[0]
        
	return sweep(G, x.real)

def temporal_cut_inv(G):
	"""
		Computes temporal cut as the smallest eigenvector of isqrt(C)Lisqrt(C)^T
		where L is the Laplacian of the temporal graph and isqrt(C) is the
		square root inverse of the Laplacian of layers that are complete graphs.
		Input:
			* G: Temporal graph
		Output:
			* temporal cut
	"""
	G.index_vertex()
	L = create_laplacian_matrix(G).todense()
	C = create_c_matrix(G).todense()
	sqrtC = sqrtm(C)
	#isqrtC = sqrtm(scipy.linalg.pinv(C))
	isqrtC = sqrtmi(C)
	M = numpy.dot(numpy.dot(isqrtC, L), isqrtC)
	
	(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(G.num_snaps(), G.num_snaps()))
	x = numpy.asarray(scipy.dot(eigvecs[:,0], isqrtC))[0]
	
	return sweep(G, x.real)

def power_method(mat, maxit):
	"""
		Simple implementation of the power method for iteratively approximating
		the largest eigenvector of a matrix
		Input:
			* mat: matrix
			* maxit: number of iterations
		Output:
			* largest eigenvector of mat
	"""
	vec = numpy.ones(mat.shape[0])
	vec = vec/numpy.linalg.norm(vec)

	for i in range(maxit):
		vec = scipy.sparse.csr_matrix.dot(vec, mat)
		vec = vec/numpy.linalg.norm(vec)
 
	return numpy.asarray(vec)

def temporal_cut_diff(G,niter=0):
	"""
		Computes temporal cut as the largest eigenvector of C-L
		where L is the Laplacian of the temporal graph and C is the
		Laplacian of a graph with complete graphs as layers.
		Input
			* G: graph
			* niter: number of iterations for power method, if 0 compute exact
		Output:
			* temporal cut
	"""
	G.index_vertex()
	L = create_laplacian_matrix(G)
	C = create_c_matrix(G)
	
	M = G.swap_cost()*C - L

	if niter == 0:
		try:
			(eigvals, eigvecs) = scipy.linalg.eigh(M.todense())
			(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1, which='LR')
			x = eigvecs[:,0]
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			Md = M.todense()
			(eigvals, eigvecs) = scipy.linalg.eigh(Md, eigvals=(G.number_of_nodes()-1,G.number_of_nodes()-1))
			x = numpy.asarray(numpy.dot(eigvecs[:,0], C.todense()))[0]
	
	else:
		x = power_method(M, niter)
	
	return sweep(G, x.real)

def independent_cuts(G):
	"""
		Computes each cut independently using eigenvector computation + sweeping
		Input:
			* Graph: Temporal graph
		Output:
			* cut
	"""
	cs = []
	G.index_vertex()
	#Computes each indepedent cut
	for t in range(G.num_snaps()):
		Lg = networkx.laplacian_matrix(G.snap(t), G.nodes())
		
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigs(Lg, k=2, which='SM')
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			Ld = Lg.todense()
			(eigvals, eigvecs) = scipy.linalg.eigh(Ld, eigvals=(0, 1))
		
		x  = eigvecs[:,numpy.argsort(eigvals)[1]].real
		c = sweep_single(G.snap(t), x, G.nodes())
		cs.append(c)
	
	#Aggregates the isoluated cuts into a temporal graph cut
	return aggreg_opt_cuts(G, cs)

def union_cut(G):
	"""
		Computes best cut for the union graph (smashing all the snapshots into a single one).
		Input:
			* Graph: Temporal graph
		Output:
			* cut
	"""
	G.index_vertex()
	#Union graph
	Gu = networkx.Graph()
	
	for v in G.nodes():
		Gu.add_node(v)

	for t in range(G.num_snaps()):
		Gs = G.snap(t)
		for e in Gs.edges():
			#Weights are summed
			if Gu.has_edge(e[0],e[1]):
				Gu.edge[e[0]][e[1]]["weight"] = Gu.edge[e[0]][e[1]]["weight"] + Gs.edge[e[0]][e[1]]["weight"]
			else:
				Gu.add_edge(e[0],e[1],weight=Gs.edge[e[0]][e[1]]["weight"])

	Lg = networkx.laplacian_matrix(Gu, G.nodes())

	try:
		#Sometimes fails with an arpack error
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(Lg, k=2, which='SM')
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		Ld = Lg.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Ld, eigvals=(0, 1))
	
	x  = eigvecs[:,numpy.argsort(eigvals)[1]].real
	c = sweep_single(Gu, x, G.nodes())
	
	#Extracting each cut
	C = numpy.array([])
	for t in range(G.num_snaps()):
		C = numpy.concatenate((C, c["cut"]), axis=0)

	c["swaps"] = 0
	c["score"] = float(c["edges"]) / (G.num_snaps() * c["size_one"] * (G.size()-c["size_one"]))
	c["cut"] = C
	c["size_one"] = c["size_one"] * G.num_snaps()
	
	return c

def iterative_cuts_cont(G, _niter):
	"""
		Searching for dynamic cuts iteratively by trying to improve
		one cut at a time. Local cuts are computed in a continuous space
		by taking into consideration the (continuous) solutions/vectors
		for previous and next cuts. Problem reduces to quadratic programming.
		Input: 
			* G: Temporal graph
			* niter: Number of iterations
		Output:
			* Temporal graph cut
	"""
	
	if _niter == 0:
		niter = G.num_snaps()
	else:
		niter = _niter

	G.index_vertex()
	#Initial solution computes best cuts for each snapshots
	cs = independent_cuts(G)
	c = G.separate_values(cs["cut"])
	score = cs["score"]
	best_score = sys.float_info.max
	best_cut = numpy.zeros(G.num_snaps() * G.size())
	
	#Computes local cuts
	for i in range(niter):
		best_score = sys.float_info.max
		for t in range(G.num_snaps()):
			if t > 0 and t < G.num_snaps()-1:
				if numpy.dot(c[t-1], c[t+1]) > numpy.dot(c[t-1], -c[t+1]):
					b = (c[t-1] + c[t+1])
				else:
					b = (c[t-1] - c[t+1])
				b = c[t-1] + c[t+1]
			elif t > 0:
				b = c[t-1]
			else:
				b = c[t+1]

			nc = local_cont_qp(G.snap(t), b)

			oc = c[t]
			c[t] = nc
			ac = G.aggregate_values(c)
			agg_cut = sweep(G, ac)
			c[t] = oc
		
			if agg_cut["score"] < best_score:
				best_cut = agg_cut
				best_score = agg_cut["score"]

		c = G.separate_values(best_cut["cut"])
	
	#Aggregates isolated cuts (already indexed according to temporal graph)
	ac = G.aggregate_values(c)
	
	return sweep(G, ac)


def iterative_cuts_disc(G, _niter):
	"""
		Searching for dynamic cuts iteratively by trying to improve
		one cut at a time. Previous and next cuts produce supernodes
		and local cuts are computed in an extended graph with 
		constraints on the assignment of supernodes (i.e. they must
		remain separate). Problem reduces to quadratic programming.
		Input: 
			* G: Temporal graph
			* niter: Number of iterations
		Output:
			* Temporal graph cut
	"""
	
	if _niter == 0:
		niter = G.num_snaps()
	else:
		niter = _niter

	G.index_vertex()
	#Initial solution computes best cuts for each snapshots
	cs = union_cut(G)
	c = G.separate_values(cs["cut"])
	score = cs["score"]
	best_score = sys.float_info.max
	best_cut = numpy.zeros(G.num_snaps() * G.size())
	
	#Computes local cuts
	for i in range(niter):
		best_score = sys.float_info.max
		for t in range(G.num_snaps()):
			if t > 0 and t < G.num_snaps()-1:
				nc = local_disc_qp(G, c[t-1], c[t+1], t)
			elif t > 0:
				nc = local_disc_qp(G, c[t-1], None, t)
			else:
				nc = local_disc_qp(G, c[t+1], None, t)
			oc = c[t]
			c[t] = nc["cut"]
			ac = G.aggregate_values(c)
			agg_cut = sweep(G, ac)
			c[t] = oc
		
			if agg_cut["score"] < best_score:
				best_cut = agg_cut
				best_score = agg_cut["score"]

		c = G.separate_values(best_cut["cut"])
	
	#Aggregates isolated cuts (already indexed according to temporal graph)
	ac = G.aggregate_values(c)
	
	return sweep(G, ac)

def local_cont_qp(G, b):
	"""
		Computes optimal cut for snapshot t given vector b which captures information
		regarding previous and next snapshots. 
		Input:
			* G: temporal graph
			* b: previous+next snapshot cut vectors
		Ouput:
			* cut vector
	"""
	#Computing all eigenvectors and eigenvalues of the Laplacian of G
	L = networkx.laplacian_matrix(G).todense()
	(eigvals, eigvecs) = scipy.linalg.eig(L)
	lamb = numpy.sort(eigvals)
	U = eigvecs[:, eigvals.argsort()]

	#Dropping smallest eigenvalue and associated eigenvector
	V = U[:,1:]
	D = numpy.diag(lamb[1:])

	#Building matrix [[D,-I],[-1/n*bb^T,D]]
	#Here be dragons
	b_ = numpy.dot(V.transpose(), b)
	M1 = scipy.sparse.block_diag((D, D))
	M2 = scipy.sparse.block_diag((scipy.sparse.identity(G.number_of_nodes()-1), (float(1.)/G.number_of_nodes())*scipy.sparse.csr_matrix(scipy.asmatrix(scipy.outer(b_, b_)))), format='csr')
	M2.indices[0:int(M2.shape[1]/2)] = M2.indices[0:int(M2.shape[1]/2)] + int(M2.shape[1]/2)
	M2.indices[int(M2.shape[1]/2):] = M2.indices[int(M2.shape[1]/2):] - int(M2.shape[1]/2)
	M = M1 - M2

	try:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1, which='SM')
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		Md = M.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Md, eigvals=(0, 0))

	lamb = eigvals[0].real

	res = numpy.dot(scipy.linalg.pinv((D-lamb*scipy.sparse.identity(G.number_of_nodes()-1))), b_)
	
	return numpy.dot(res, V.transpose()).real

def local_disc_qp(G, Cb, Ca, t):
	"""
		Computes optimal cut for snapshot t given cuts Cb and Ca. Order of Cb
		Input:
			* G: temporal graph
			* Cb: previous cut
			* Ca: next cut (can be None)
			* t: snapshot
		Ouput:
			* cut vector
	"""
	Gl = G.snap(t).copy()
	
	#Creates extended graph (with supernodes)
	#Nodes are indexed using temporal graph index
	Gl.add_node('vb1')
	Gl.add_node('vb2')
	i = 0
	for v in G.nodes():
		if Cb[i] < 0:
			Gl.add_edge(v,'vb1', weight=G.swap_cost())
		else:
			Gl.add_edge(v,'vb2', weight=G.swap_cost())
		i = i + 1
	
	if Ca is not None:
		Gl.add_node('va1')
		Gl.add_node('va2')
		i = 0
		for v in G.nodes():
			if Ca[i] < 0:
				Gl.add_edge(v,'va1', weight=G.swap_cost())
			else:
				Gl.add_edge(v,'va2', weight=G.swap_cost())
			i = i + 1
	
	vb1 = Gl.nodes().index('vb1')
	vb2 = Gl.nodes().index('vb2')

	#Creates matrix with constraints (weights of supernodes on the same snapshot add to 0)
	if Ca is None:
		C = create_constraint_matrix(Gl.number_of_nodes(), vb1, vb2)
	else:
		va1 = Gl.nodes().index('va1')
		va2 = Gl.nodes().index('va2')
	
		C = create_constraint_matrix(Gl.number_of_nodes(), vb1, vb2, va1, va2)
	
	#Solution based on chapter 12 quadratic optimization problems.
	#Maximizing quadratic functions on the unit sphere
	
	#QR depcomposition
	Q, R = scipy.linalg.qr(C.transpose().todense(), mode='full', pivoting=False)
	L = networkx.laplacian_matrix(Gl)
	
	sQ = scipy.sparse.csr_matrix(Q)
	
	QLQ = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(sQ.transpose(), L), sQ)

	if Ca is not None:
		G22 = QLQ[3:, 3:]
	else:
		G22 = QLQ[2:, 2:]
	
	#Solution via eigenvalue problem
	try:
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(G22, k=1, which='SM') 
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		G22D = G22.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(G22D, eigvals=(0, 1))
	
	z = eigvecs[:,0]
	sign = 1
	if Ca is None:
		#Projecting solution back to original space
		x = scipy.sparse.csr_matrix.dot(sQ, numpy.concatenate((numpy.zeros(2), z)))
		if x[vb1] < 0:
			x[vb1] = -sys.float_info.max
			x[vb2] = sys.float_info.max
		else:
			x[vb1] = sys.float_info.max
			x[vb2] = -sys.float_info.max
			sign = -1
			
	else:
		#Projecting solution back to original space
		x = scipy.dot(Q, numpy.concatenate((numpy.zeros(3), z)))
		if x[vb1] < 0:
			x[vb1] = -sys.float_info.max
			x[vb2] = sys.float_info.max
		else:
			x[vb1] = sys.float_info.max
			x[vb2] = -sys.float_info.max
			sign = -1
		
		if x[va1] < 0:
			x[va1] = -sys.float_info.max
			x[va2] = sys.float_info.max
		else:
			x[va1] = sys.float_info.max
			x[va2] = -sys.float_info.max

	#Sweeping
	if Ca is None:
		c = sweep_single_qp(Gl, x.real, True)
	else:
		c = sweep_single_qp(Gl, x.real, False)
	
	#Cut has to be projected using proper indexing
	C = {}
	C["cut"] = numpy.zeros(G.size())
	j = 0
	
	for v in Gl.nodes():
		if v != "va1" and v != "va2" and v != "vb1" and v != "vb2":
			i = G.index_vertex(t, v)
			p,t = G.rev_index_pos_single(i)
			C["cut"][p] = sign * c["cut"][j]
		j = j + 1

	C["edges"] = c["edges"]

	return C

class IndependentCuts(object):
	"""
		Wrapper class for independent cuts
	"""
	def __init__(self, name):
		self._name = name

	def name(self):
		return self._name

	def cut(self, G):
		return independent_cuts(G)

class UnionCut(object):
	"""
		Wrapper class for union cut
	"""
	def __init__(self, name):
		self._name = name
	
	def name(self):
		return self._name

	def cut(self, G):
		return independent_cuts(G)

class TemporalCutProd(object):
	"""
		Wrapper class for temporal cut product
	"""
	def __init__(self, name):
		self._name = name
	
	def name(self):
		return self._name

	def cut(self, G):
		return temporal_cut_prod(G)

class TemporalCutInv(object):
	"""
		Wrapper class for temporal cut inverse
	"""
	def __init__(self, name):
		self._name = name
	
	def name(self):
		return self._name

	def cut(self, G):
		return temporal_cut_inv(G)

class TemporalCutDiff(object):
	"""
		Wrapper class for temporal cut difference
	"""
	def __init__(self, name, niter=0):
		self._name = name
		self.niter = niter
	
	def name(self):
		return self._name

	def cut(self, G):
		return temporal_cut_diff(G, self.niter)

class IterativeCutsDisc(object):
	"""
		Wrapper class for discrete iterative cuts
	"""
	def __init__(self, name, niter=0):
		self._name = name
		self.niter = niter
	
	def name(self):
		return self._name

	def cut(self, G):
		return iterative_cuts_disc(G, self.niter)

class IterativeCutsCont(object):
	"""
		Wrapper class for continuous iterative cuts
	"""
	def __init__(self, name, niter=0):
		self._name = name
		self.niter = niter
	
	def name(self):
		return self._name

	def cut(self, G):
		return iterative_cuts_cont(G, self.niter)

