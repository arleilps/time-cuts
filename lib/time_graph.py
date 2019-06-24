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
from numpy.linalg import eigh
from heapq import heappush, heappop

class TimeGraph(object):
	"""
		Class for managing temporal/dynamic graphs
	"""
	def __init__(self, swp_cost=1., vertex_swp_cost='uniform'):
		"""
			Initialization.
			Input:
				* swp_cost: cost of swaping a node across cuts
		"""
		self.graphs = [networkx.Graph()]
		self.swp_cost = swp_cost
		self.idx_vertex = []
		self._node_list = []
		self._connect_weight = 0.
		self._active_vertex = {}
		self._number_of_nodes = 0
		self._number_of_nodes_snap = [0]
		self._swp_cost_vertex = {}
		self.vertex_swp_cost = vertex_swp_cost
		self.max_swp_cost = swp_cost

	def set_swap_cost(self, swp_cost):
		self.swp_cost = swp_cost
		self.max_swp_cost = swp_cost
		self.set_swap_cost_vertices()

	def merge_cuts(self, cuts, graphs):
		IDs = [-1., 1., 2., 3., 4., 5., 6., 7.]
		merged = numpy.zeros(self.number_of_nodes())
		
		ID = 0
		for c in range(0, len(cuts)):
			for i in range(cuts[c].shape[0]):
				if cuts[c][i] != 0.:
					(v,t) = graphs[c].rev_index_pos(i)
					j = self.index_vertex(t,v)
					
					if cuts[c][i] < 0:
						merged[j] = IDs[ID]
					else:
						merged[j] = IDs[ID+1]

			ID = ID + 2

		return merged

	def activate(self, v, t):
		if v not in self._active_vertex:
			self._active_vertex[v] = {}

		self._active_vertex[v][t] = True
		self._number_of_nodes = self._number_of_nodes + 1
		self._number_of_nodes_snap[t] = self._number_of_nodes_snap[t] + 1

	def deactivate(self, v, t):
		self._active_vertex[v][t] = False
		self._number_of_nodes = self._number_of_nodes - 1
		self._number_of_nodes_snap[t] = self._number_of_nodes_snap[t] - 1

	def active_nodes(self):
		node_list = []
		for t in range(self.num_snaps()):
			for v in self.nodes():
				if self.active(v,t):
					node_list.append((v,t))

		return node_list

	def active_degree(self, v, t):
		deg = 0.
		if self.active(v,t):
			for u in self.graphs[t].neighbors(v):
				if self.active(u,t):
					deg =  deg + self.graphs[t][v][u]['weight']
		return deg

	def active(self, v, t):
		return self._active_vertex[v][t]

	def copy(self):
		nG = TimeGraph(self.swp_cost)	
		
		for t in range(self.num_snaps()):
			G = self.snap(t)
			for e in G.edges():
				v1 = e[0]
				v2 = e[1]
				nG.add_edge(v1, v2, t, G[v1][v2]['weight'])
		nG.index_vertex()

		for t in range(self.num_snaps()):
			for v in self.nodes():
				c = self.swap_cost_vertex(v,t)
				nG.set_swap_cost_vertex(v, t, c)
		
		return nG

	def break_graph_cut(self, cut):
		G1 = self.copy()
		G2 = self.copy()

		for t in range(self.num_snaps()):
			for v in self.nodes():
				if self.active(v,t):
					i = self.index_vertex(t,v)
					if cut[i] < 0:
						G2.deactivate(v,t)
					else:
						G1.deactivate(v,t)
				else:	
					G1.deactivate(v,t)
					G2.deactivate(v,t)
		
		return G1, G2

	def nodes(self):
		return self._node_list

	def number_of_nodes(self, t=None):
		if t is None:
			return self._number_of_nodes
		else:
			return self._number_of_nodes_snap[t]

	def number_of_edges(self):
		n = 0
		for t in range(len(self.graphs)):
			n = n + self.graphs[t].number_of_edges()

		return n

	def set_swap_cost_vertices(self):
		if self.vertex_swp_cost == 'random':
			for v in self.nodes():
				if v not in self._swp_cost_vertex:
					self._swp_cost_vertex[v] = {}

				for t in range(len(self.graphs)-1):
					self._swp_cost_vertex[v][t] = self.swp_cost * numpy.random.random()
					if self._swp_cost_vertex[v][t] > self.max_swp_cost:
							self.max_swp_cost = self._swp_cost_vertex[v][t]

		elif self.vertex_swp_cost == 'degree':
			for v in self.nodes():
				if v not in self._swp_cost_vertex:
					self._swp_cost_vertex[v] = {}
				
				for t in range(len(self.graphs)):
					self._swp_cost_vertex[v][t] = self.swp_cost + self.graphs[t].degree(v, weight='weight')
					if self._swp_cost_vertex[v][t] > self.max_swp_cost:
						self.max_swp_cost = self._swp_cost_vertex[v][t]
		elif self.vertex_swp_cost == 'uniform':
			for v in self.nodes():
				if v not in self._swp_cost_vertex:
					self._swp_cost_vertex[v] = {}
				
				for t in range(len(self.graphs)):
					self._swp_cost_vertex[v][t] = self.swp_cost 
					if self._swp_cost_vertex[v][t] > self.max_swp_cost:
						self.max_swp_cost = self._swp_cost_vertex[v][t]
			

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
			self.set_swap_cost_vertices()

		if t is not None and v is not None:	
			return self.idx_vertex[t][v]

	def make_connected(self, _connect_weight=0.001):
		self._connect_weight = _connect_weight
		
		if _connect_weight > 0.:
			for t in range(len(self.graphs)):
				for v in self.graphs[t].nodes():
					if self.graphs[t].degree(v, 'weight') == 0:
						for u in self.graphs[t].nodes():
							if v != u:
								self.graphs[t].add_edge(u,v,weight=_connect_weight, timestamp=-1)
	def connect_weight(self):
		return self._connect_weight

	def swap_cost_vertex(self, v=None, t=None):
		"""
			Returns the swap cost (constant)
		"""
		if v is None or v not in self._swp_cost_vertex:
			return self.swp_cost
		else:
			return self._swp_cost_vertex[v][t]

	def max_swap_cost(self):
		return self.max_swp_cost

	def set_swap_cost_vertex(self, v, t, cost):
		if v not in self._swp_cost_vertex:
			self._swp_cost_vertex[v] = {}

		self._swp_cost_vertex[v][t] = cost

	def extend(self, t):
		"""
			Extends the temporal graph up to t timestamps
			Input:
				* t: timestamp for extension
		"""
		while len(self.graphs) <= t:
			self.graphs.append(networkx.Graph())
			self._number_of_nodes_snap.append(0)

			for v in self.graphs[0].nodes():
				self.graphs[-1].add_node(v)
				self.activate(v, len(self.graphs)-1)
					
	def add_edge(self, v1,v2,t,w=1.,tsp=0):
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
												   
		self.graphs[t].add_edge(v1, v2, weight=w, timestamp=tsp)

	def add_node(self, v):
		"""
			Adds node to temporal graph.
			Input:
				* v: node
		"""
		for t in range(len(self.graphs)):
			if v not in self.graphs[t]:
				self.graphs[t].add_node(v)
				self.activate(v, t)

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

def sweep_sparse_laplacian(G, x):
	"""
		Eigenvector sweep rounding on a single graph/snapshot.
		Input:
			* G: graph
			* x: vector
		Output:
			* rounded vector
	"""
	best_val = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	nodes_one = {}
	best_edges_cut = 0

	for i in range(x.shape[0]):
		size_one = size_one + 1
		(u,t) = G.rev_index_pos(sorted_x[i])
		
		nodes_one[(u,t)] = True
		
		for v in G.snap(t).neighbors(u):
			if (v,t) not in nodes_one:
				edges_cut = edges_cut + G.snap(t)[v][u]["weight"]
			else:
				edges_cut = edges_cut - G.snap(t)[v][u]["weight"]

		if t > 0:
			if (u,t-1) not in nodes_one:
				edges_cut = edges_cut + G.swap_cost_vertex(u,t-1)
			else:
				edges_cut = edges_cut - G.swap_cost_vertex(u,t-1)
				
		if t < G.num_snaps()-1:
			if (u,t+1) not in nodes_one:
				edges_cut = edges_cut + G.swap_cost_vertex(u,t)
			else:
				edges_cut = edges_cut - G.swap_cost_vertex(u,t)

			
		den = size_one * (G.number_of_nodes()-size_one)

		if den > 0:
			val = float(edges_cut) / den
		else:
			val = G.number_of_nodes()

		if val <= best_val:
			best_cand = i
			best_val = val
			best_edges_cut = edges_cut

	vec = []

	vec = numpy.zeros(G.number_of_nodes())

	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return vec

def sweep_norm_laplacian(G, x):
	"""
		Eigenvector sweep rounding on a single graph/snapshot.
		Input:
			* G: graph
			* x: vector
		Output:
			* rounded vector
	"""
	best_val = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	deg_one = {}
	best_edges_cut = 0
	best_deg_one = 0
	vol = G.volume()

	for i in range(x.shape[0]):
		(u,t) = G.rev_index_pos(sorted_x[i])
		deg_one = deg_one + G.degree(u,t)
		nodes_one[(u,t)] = True
		
		for v in G.snap(t).neighbors(u):
			if (v,t) not in nodes_one:
				edges_cut = edges_cut + G.snap(t)[v][u]["weight"]
			else:
				edges_cut = edges_cut - G.snap(t)[v][u]["weight"]

		if t > 0:
			if (u,t-1) not in nodes_one:
				edges_cut = edges_cut + G.swap_cost_vertex(u,t-1)
			else:
				edges_cut = edges_cut - G.swap_cost_vertex(u,t-1)
				
		if t < G.num_snaps()-1:
			if (u,t+1) not in nodes_one:
				edges_cut = edges_cut + G.swap_cost_vertex(u,t)
			else:
				edges_cut = edges_cut - G.swap_cost_vertex(u,t)

			
		den = deg_one * (vol-deg_one)

		if den > 0:
			val = float(edges_cut) / den
		else:
			val = G.number_of_nodes()

		if val <= best_val:
			best_cand = i
			best_val = val

	vec = []

	vec = numpy.zeros(G.number_of_nodes())

	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return vec

def sweep_norm_single(G, x, nodes_list=None):
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
	deg_one = 0
	edges_cut = 0
	nodes_one = {}
	vol = 0.
	for v in G.nodes():
		vol = vol + G.degree(v, weight='weight')

	if nodes_list is None:
		nodes_list = G.nodes()

	for i in range(x.shape[0]-1):
		deg_one = deg_one + G.degree(nodes_list[sorted_x[i]], weight='weight')
		
		nodes_one[nodes_list[sorted_x[i]]] = True
		u = nodes_list[sorted_x[i]]
		
		for v in G.neighbors(u):
			if v not in nodes_one:
				edges_cut = edges_cut + G[v][u]["weight"]
			else:
				edges_cut = edges_cut - G[v][u]["weight"]
			
		den = deg_one * (vol-deg_one)

		if den > 0:
			val = float(edges_cut) / den
		else:
			val = networkx.number_of_nodes(G)

		if val <= best_val:
			best_cand = i
			best_val = val
			best_edges_cut = edges_cut

	vec = []

	vec = numpy.zeros(networkx.number_of_nodes(G))

	for i in range(x.shape[0]):
		if i <= best_cand:
			vec[sorted_x[i]] = -1.
		else:
			vec[sorted_x[i]] = 1.

	return {"cut": vec, "edges": best_edges_cut}

def sweep_sparse_single(G, x, nodes_list=None):
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

def sweep_norm(G, x):
	"""
				Eigenvector sweep rounding for temporal graph.
				Input:
						* G: graph
						* x: vector
						* nodes_list: list of nodes (order), G.nodes() is applied if None
				Output:
						* rounded vector
	"""
	best_score = sys.float_info.max
	sorted_x = numpy.argsort(x)
	size_one = 0
	edges_cut = 0
	swaps = 0
	nodes_one = []
	sizes_one = []
	deg_one = []
	volumes = []
	den = 0
	degrees = {}

	for t in range(G.num_snaps()):
		nodes_one.append({})
		sizes_one.append(0)
		deg_one.append(0)

		deg = 0
		degrees[t] = {}

		for v in G.nodes():
			if G.active(v, t):
				d = G.active_degree(v, t)
				deg = deg + d

				degrees[t][v] = d

		volumes.append(deg)

	for i in range(x.shape[0]-1):
		(v,t) = G.rev_index_pos(sorted_x[i])
		if G.active(v, t):
			den = den - ((volumes[t] - deg_one[t]) * deg_one[t])
			sizes_one[t] = sizes_one[t] + 1
			deg_one[t] = deg_one[t] + G.active_degree(v,t)
			den = den + ((volumes[t] - deg_one[t]) * deg_one[t])
			nodes_one[t][v] = True
		
			for u in G.graphs[t].neighbors(v):
				if G.active(u, t):
					if u not in nodes_one[t]:
						edges_cut = edges_cut + G.graphs[t][v][u]["weight"] 
					else:
						edges_cut = edges_cut - G.graphs[t][v][u]["weight"]
	
			if t+1 < G.num_snaps():
				if G.active(v, t+1):
					if v not in nodes_one[t+1]:
						swaps = swaps + G.swap_cost_vertex(v, t)
					else:
						swaps = swaps - G.swap_cost_vertex(v, t) 

			if t > 0:
				if G.active(v, t-1):
					if v not in nodes_one[t-1]:
						swaps = swaps + G.swap_cost_vertex(v, t-1) 
					else:
						swaps = swaps - G.swap_cost_vertex(v, t-1) 
		
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
		(v,t) = G.rev_index_pos(sorted_x[i])
		
		if G.active(v,t):
			if i <= best:
				vec[sorted_x[i]] = -1.
			else:
				vec[sorted_x[i]] = 1.
		else:
			vec[sorted_x[i]] = 0.

	if best_score < 0:
		best_score = 0.

	return {"cut": vec, "score": best_score, "edges": best_edges_cut, "swaps": best_swaps}

def sweep_sparse(G, x):
	"""
				Eigenvector sweep rounding for temporal graph.
				Input:
						* G: graph
						* x: vector
						* nodes_list: list of nodes (order), G.nodes() is applied if None
				Output:
						* rounded vector
	"""
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
		if G.active(v, t):
			den = den - sizes_one[t] * (G.number_of_nodes(t) - sizes_one[t])
			sizes_one[t] = sizes_one[t] + 1
			den = den + sizes_one[t] * (G.number_of_nodes(t) - sizes_one[t])

			nodes_one[t][v] = True
			
			for u in G.graphs[t].neighbors(v):
				if u not in nodes_one[t]:
					edges_cut = edges_cut + G.graphs[t][v][u]["weight"]
				else:
					edges_cut = edges_cut - G.graphs[t][v][u]["weight"]

			if t+1 < G.num_snaps():
				if v not in nodes_one[t+1]:
					swaps = swaps + G.swap_cost_vertex(v, t)
				else:
					swaps = swaps - G.swap_cost_vertex(v, t)

			if t > 0:
				if v not in nodes_one[t-1]:
					swaps = swaps + G.swap_cost_vertex(v, t-1)
				else:
					swaps = swaps - G.swap_cost_vertex(v, t-1)
		
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
	
	if best_score < 0:
		best_score = 0.

	for i in range(x.shape[0]):
		(v,t) = G.rev_index_pos(sorted_x[i])
		
		if G.active(v,t):
			if i <= best:
				vec[sorted_x[i]] = -1.
			else:
				vec[sorted_x[i]] = 1.
		else:
			vec[sorted_x[i]] = 0.

	return {"cut": vec, "score": best_score, "edges": best_edges_cut, "swaps": best_swaps}

def evaluate_cut(G, x, norm=False):
	"""
				Compute cut measures for a given cut.
				Input:
						* G: graph
						* x: vector
				Output:
						* edges_cut: number of edges cut
						* swaps: number of node swaps
						* den: product |X||V-X|
	"""
	edges_cut = 0
	swaps = 0
	den = 0
	
	if norm is True:
		volumes = []
		for t in range(G.num_snaps()):
			deg = 0

			for v in G.nodes():
				d = G.active_degree(v, t)
				deg = deg + d
	
			volumes.append(deg)

	for t in range(G.num_snaps()):
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]

			if G.active(v1, t) and G.active(v2, t):			
				cut = x[G.index_vertex(t, v1)] + x[G.index_vertex(t, v2)]

				if abs(cut) < 1.:
					edges_cut = edges_cut + G.snap(t)[v1][v2]["weight"] 

	for t in range(G.num_snaps()):
		size_one = 0
		deg_one = 0
		for v in G.snap(t).nodes():
			if t < G.num_snaps()-1:
				if G.active(v, t) and G.active(v, t+1):
					cut = x[G.index_vertex(t, v)] + x[G.index_vertex(t+1, v)]
			
					if abs(cut) < 1.:
						swaps = swaps + G.swap_cost_vertex(v,t)
			
			if G.active(v,t):
				if x[G.index_vertex(t, v)] < 0:
					size_one = size_one + 1
					if norm is True:
						deg_one = deg_one + G.active_degree(v,t)

		if norm is True:
			den = den + (volumes[t]-deg_one) * deg_one
		else:
			den = den + size_one * (G.number_of_nodes(t) - size_one)
	
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

	for i in range(1, len(p)-1):
		if p[i] % 2 == 0:
			c[i-1]["cut"] =  -1 * c[i-1]["cut"]
		
		vec = numpy.concatenate((vec, c[i-1]["cut"]), axis=0)

	return vec

def create_laplacian_matrix(G, norm=False):
	"""
		Creates laplacian matrix for temporal graph.
		This is the laplacian of the layered/stacked graph with edges
		connecting the same node within consecutive snapshots
	"""
	row = []
	column = []
	value = []

	if norm is True:
		Deg = create_degree_matrix(G)
	else:
		Deg = scipy.sparse.identity(G.size() * G.num_snaps(), dtype=float,format='csr')

	for t in range(G.num_snaps()):
		Ag = networkx.adjacency_matrix(G.snap(t), G.nodes(), weight='weight')
		for (i,j) in zip(*scipy.nonzero(Ag)):
			
			(vi, ti) = G.rev_index_pos(G.size()*t + i)
			(vj, tj) = G.rev_index_pos(G.size()*t + j)

			if G.active(vi, t) and G.active(vj, t):
				row.append(G.size()*t + i)
				column.append(G.size()*t + j)

				if Deg[G.size()*t + i, G.size()*t + i] > 0 and Deg[G.size()*t + j, G.size()*t + j] > 0:
					value.append(Ag[i,j] / math.sqrt(Deg[G.size()*t + i, G.size()*t + i]*Deg[G.size()*t + j, G.size()*t + j]))
				
	for t in range(G.num_snaps()-1):
		for v in range(G.size()):
			
			(vi, ti) = G.rev_index_pos(v)
			
			if G.active(vi, t) and G.active(vi,t+1):
				row.append(t*G.size() + v)
				column.append((t+1)*G.size() + v)
				value.append(G.swap_cost_vertex(vi,t))
			
				column.append(t*G.size() + v)
				row.append((t+1)*G.size() + v)
				value.append(G.swap_cost_vertex(vi,t))

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

	if norm is True:
		return D-A
	else:
		return D-A

def create_degree_matrix(G):
	row = []
	column = []
	value = []

	for t in range(G.num_snaps()):
		for v in G.nodes():
			if G.active(v,t):
				deg = 0.
				vi = G.index_vertex(t,v)
				
				for u in G.snap(t).neighbors(v):
					if G.active(u,t):
						deg = deg + G.snap(t)[v][u]['weight']

				if deg > 0.:
					row.append(vi)
					column.append(vi)
					value.append(deg)
	
	sz = G.num_snaps() * G.size()

	return scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

def sqrtmi_degree_matrix(D):
	row = []
	column = []
	value = []
	
	for (i,j) in zip(*scipy.nonzero(D)):
		row.append(i)
		column.append(j)
		value.append(1./math.sqrt(D[i,j]))

	return scipy.sparse.csr_matrix((value, (row, column)), shape=D.shape, dtype=float)

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
				(vi, ti) = G.rev_index_pos(i)
				(vj, tj) = G.rev_index_pos(j)
				
				if i != j:
#					if G.active(vi, t) and G.active(vj, t):
					row.append(t*G.size() + i)
					column.append(t*G.size() + j)
					value.append(1.)
	
	sz = G.num_snaps() * G.size()
	
	A = scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)
	
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

def prod_cut(G, norm=False):
	"""
		Computes temporal cut via eigenvectors of CLC^T matrix, where L is the Laplacian
		of the temporal graph and C is the Laplacian of layers that are complete graphs.
		Input: 
			* G: Temporal graph
		Output:
			* temporal cut
	"""
	G.index_vertex()

	if norm is True:
		L = create_laplacian_matrix(G, norm=True)
	else:
		L = create_laplacian_matrix(G, norm=False)
	
	C = create_c_matrix(G)
	
	M = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(C, L), C)
	n_removed = G.size()*G.num_snaps() - G.number_of_nodes()
	
	if n_removed == 0:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(M, k=G.num_snaps()+1, which='SA')
			x = eigvecs[:,-1]
		except:
			try:
				(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=G.num_snaps()+1, which='SR')
				idx = eigvals.argsort()[::-1]
				eigvals = eigvals[idx]
				eigvecs = eigvecs[:,idx]
				x = eigvecs[:,0]
			except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
				Md = M.todense()
				(eigvals, eigvecs) = scipy.linalg.eigh(Md, eigvals=(G.num_snaps(), G.num_snaps()))
				x = eigvecs[:,0]
	else:
		Md = M.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Md, eigvals=(G.num_snaps()+n_removed, G.num_snaps()+n_removed))
		x = numpy.asarray(eigvecs[:,0])
	
	if norm is True:
#		x = scipy.sparse.csr_matrix.dot(x, sqrtiD)
		return sweep_norm(G, x.real), x.real
	else:
		return sweep_sparse(G, x.real), x.real

def inv_cut(G, norm=False):
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
	if norm is True:
		L = create_laplacian_matrix(G, norm=True)
		L = L.todense()
	else:
		L = create_laplacian_matrix(G).todense()

	C = create_c_matrix(G).todense()
	isqrtC = sqrtmi(C)
	M = numpy.dot(numpy.dot(isqrtC, L), isqrtC)
	
	#(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(G.num_snaps()+(G.size()*G.num_snaps() - G.number_of_nodes()), G.num_snaps()+(G.size()*G.num_snaps() - G.number_of_nodes())))
	(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=(2, 2))

	x = eigvecs[:, 0]

	if norm is True:
		return sweep_norm(G, x.real)
	else:
		return sweep_sparse(G, x.real)


def power_method(mat, init=None, eps=1.0e-1):
	"""
		Simple implementation of the power method for iteratively approximating
		the largest eigenvector of a matrix
		Input:
			* mat: matrix
			* maxit: number of iterations
		Output:
			* largest eigenvector of mat
	"""
	if init is None:
		init = numpy.random.random((mat.shape[0],1))
	else:
		init = numpy.array([init]).T

	vec = scipy.sparse.csr_matrix.dot(mat, init)
	vec = vec/numpy.linalg.norm(vec)

	while numpy.linalg.norm(vec - init) > eps: 
		init = vec
		vec = scipy.sparse.csr_matrix.dot(mat, init)
		vec = vec/numpy.linalg.norm(vec)

	return numpy.asarray(vec)[:,0]

def laplacian_cut(G, norm=False):
	"""
	"""
	G.index_vertex()

	if norm is True:
		L = create_laplacian_matrix(G,norm=True)
	else:
		L = create_laplacian_matrix(G)
	
	n_removed = G.size()*G.num_snaps() - G.number_of_nodes()

	if n_removed == 0:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(L, k=2, which='SA')
			x = eigvecs[:,1+n_removed]
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			try:
				(eigvals, eigvecs) = scipy.sparse.linalg.eigs(L, k=2, which='SR')
				idx = eigvals.argsort()[::-1]
				eigvals = eigvals[idx]
				eigvecs = eigvecs[:,idx]
				x = eigvecs[:,0]
			except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
				L = L.todense()
				(eigvals, eigvecs) = scipy.linalg.eigh(L, eigvals=(1,1))
				x = eigvecs[:,0]
	else:
		L = L.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(L, eigvals=(1+n_removed,1+n_removed))
		x = eigvecs[:,0]

	if norm is True:
		return sweep_norm(G, x.real)
	else:
		return sweep_sparse(G, x.real)

def diff_cut(G,norm=False,eps=0.):
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

	if norm is True:
		L = create_laplacian_matrix(G, norm=True)
	else:
		L = create_laplacian_matrix(G)
		
	C = create_c_matrix(G)
	
	M = 3. * (2.*G.max_swap_cost() + G.size()) * C - L
	
	init = union_vec(G, norm)
	
	if eps == 0.:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(M, k=1, which='LA', v0=init)
			x = eigvecs[:,0]
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			try:
				(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1, which='LR', v0=init)
				x = numpy.asarray(eigvecs[:,0])
			except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
				M = M.todense()
				(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=((G.size() * G.num_snaps())-1,(G.size() * G.num_snaps())-1))
				x = eigvecs[:,0]
	else:
		x = power_method(M, init, eps)

	if norm is True:
		return sweep_norm(G, x.real)
	else:
		return sweep_sparse(G, x.real)

def get_partitions_tree_rec(t, G, partitions):
	if t.data is None:
		for c in t.children:
			get_partitions_tree_rec(c, G, partitions)
	else:
		partitions.append(t.data)
				    
def get_partitions_tree(t, G):
	partitions = []
	get_partitions_tree_rec(t, G, partitions)
					        
	return partitions
						    
def get_partition_assign(G, partitions):
	assign = numpy.zeros(G.size() * G.num_snaps())
							    
	for p in range(len(partitions)):
		for i in range(len(partitions[p])):
			assign[partitions[p][i]] = p
										                
	return assign

def multi_cut_hierarchy(G, K, eps=0., norm=False):
	G.index_vertex()
	root = Node(None)
	k = 1
	cand_cuts = []
	cut = diff_cut(G, norm, eps)
	cut["parent"] = root
	cut["graph"] = G

	cand_cuts.append(cut)
	
	while k < K and len(cand_cuts) > 0:
		best_cut = None
		b = 0
		for c in range(len(cand_cuts)):
			if best_cut is None or cand_cuts[c]["score"] < best_cut["score"]:
				best_cut = cand_cuts[c]
				b = c

		(G1,G2) = best_cut["graph"].break_graph_cut(best_cut["cut"])
		
		if G1.number_of_nodes() == 1:
			(v, t) = G1.active_nodes()[0]
			i = G.index_vertex(t, v)
			n = Node([i])
			best_cut["parent"].add_child(n)
		else:
			n = Node(None)
			cut = diff_cut(G1, norm, eps)
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
			cut = diff_cut(G2, norm, eps)
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

	return get_partition_assign(G, get_partitions_tree(root, G))

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

def temporal_modularity(G, assign, omega):
	M = create_modularity_matrix(G, omega)
	modularity = 0.	
	for (i,j) in zip(*scipy.nonzero(M)):
		if assign[i] == assign[j]:
			modularity = modularity + M[i,j]

	return modularity

def evaluate_multi_cut(G, assign, omega):
	n_swaps = 0.
	cut = 0.
	balance = 0.
	norm_balance = 0.

	for t in range(G.num_snaps()):
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]
			i1 = G.index_vertex(t,v1)
			i2 = G.index_vertex(t,v2)
			
			if assign[i1] != assign[i2]:
				cut = cut + G.snap(t)[v1][v2]["weight"]

	for t in range(G.num_snaps()-1):
		for v in G.snap(t).nodes():
			i1 = G.index_vertex(t,v)
			i2 = G.index_vertex(t+1,v)
			
			if assign[i1] != assign[i2]:
				n_swaps = n_swaps + G.swap_cost_vertex(v,t)

	for t in range(G.num_snaps()-1):
		partitions = {}
		norm_partitions = {}

		for v in G.snap(t).nodes():
			i1 = G.index_vertex(t,v)
			p = assign[i1]

			if p not in partitions:
				partitions[p] = 0
				norm_partitions[p] = 0

			norm_partitions[p] = norm_partitions[p] + G.active_degree(v,t)
			partitions[p] = partitions[p] + 1
		
		prod = 1.
		norm_prod = 1.
		for p in partitions:
			prod = prod * partitions[p]
			norm_prod = norm_prod * norm_partitions[p]

		balance = balance + prod
		norm_balance = norm_balance + norm_prod

	ratio = float(cut + n_swaps) / balance
	norm_ratio = float(cut + n_swaps) / norm_balance

	res = {}
	res["Sparsity"] = ratio
	res["N-sparsity"] = norm_ratio
	res["Cut"] = cut
	res["Modularity"] = temporal_modularity(G, assign, omega)

	return res

def create_sparse_block_matrix(Us):
	"""
		Creates a sparse matrix with eigenvector matrices of snapshots/layers in the main (block) diagonal.
		Input
			* Us: List with eigenvector matrices (sorted by snapshot)
		Output:
			* sparse eigenvector matrix
	"""
	row = []
	column = []
	value = []
	
	for t in range(len(Us)):
		for i in range(Us[t].shape[0]):
			for j in range(Us[t].shape[1]):
				row.append(t*Us[t].shape[0] + i)
				column.append(t*Us[t].shape[0] + j)
				value.append(Us[t][i][j])
	
	sz = len(Us) * Us[0].shape[0]
	
	return scipy.sparse.csr_matrix((value, (row, column)), shape=(sz, sz), dtype=float)

def create_sparse_block_swap_matrix(G):
	"""
	"""
	row = []
	column = []
	value = []
	
	for t in range(G.num_snaps()-1):
		for v in range(G.size()):
			
			(vi, ti) = G.rev_index_pos(v)
			
			if G.active(vi, t) and G.active(vi,t+1):
				row.append(t*G.size() + v)
				column.append((t+1)*G.size() + v)
				value.append(G.swap_cost_vertex(vi,t))
			
				column.append(t*G.size() + v)
				row.append((t+1)*G.size() + v)
				value.append(G.swap_cost_vertex(vi,t))

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

def find_constant_eigenvector(eigvecs, eigvals):
	eigvecs_sum = eigvecs.sum(axis=0)
	return numpy.argmax(numpy.absolute(eigvecs_sum))
	
def fast_cut(G,k,norm=False,eps=1.0e-5):
	"""
		Fast version of temporal cut based on largest eigenvector of C-L
		where L is the Laplacian of the temporal graph and C is the
		Laplacian of a graph with complete graphs as layers. The algorithm follows 
		divide-and-conquer. In the divide phase, the spectrum of each Laplacian matrix
		(for each snapshot/layer) are computed and low-rank approximations of the
		eigenvector matrices are kept. In the conquer phase, the largest eigenvector
		of C-L is approximated using a large sparse matrix with low-rank representations
		of the eigenvector matrices.
		Input
			* G: graph
			* niter: number of iterations for power method, if 0 compute exact
		Output:
			* temporal cut
	"""
	G.index_vertex()
	#divide
	Us = []
	
	lambs = numpy.array([])
	for t in range(G.num_snaps()):
		if norm is True:
			L = networkx.normalized_laplacian_matrix(G.snap(t), G.nodes(), weight='weight')
		else:
			L = networkx.laplacian_matrix(G.snap(t), G.nodes(), weight='weight')

		connected = networkx.is_connected(G.snap(t))
			
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(L, k=k, which='SA')
			eigvals = numpy.concatenate((3*(G.size()+2.*G.max_swap_cost()) - eigvals, G.size() * numpy.zeros(G.size()-k)), axis=0)
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			try:
				(eigvals, eigvecs) = scipy.sparse.linalg.eigs(L, k=k, which='SR')
				idx = eigvals.argsort()
				eigvals = eigvals[idx]
				eigvecs = eigvecs[:,idx]
				eigvals = numpy.concatenate((3*(G.size()+2.*G.max_swap_cost()) - eigvals, G.size() * numpy.zeros(G.size()-k)), axis=0)
			except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
				L = L.todense()
				(eigvals, eigvecs) = scipy.linalg.eigh(L, eigvals=(0,k-1))
				eigvals = numpy.concatenate((3*(G.size()+2.*G.max_swap_cost()) - eigvals, numpy.zeros(G.size()-k)), axis=0)
		
		if connected:
			const_eig = 0
		else:
			const_eig = find_constant_eigenvector(eigvecs, eigvals)
		
		eigvals[const_eig] = 0.
		Us.append(eigvecs)

		lambs = numpy.concatenate((lambs, eigvals), axis=0)
	
	#conquer
	lamb = scipy.sparse.diags(lambs, 0, format='csr')
	U = create_sparse_block_matrix(Us)
	B = create_sparse_block_swap_matrix(G)
	B = scipy.sparse.csr_matrix.dot(scipy.sparse.csr_matrix.dot(U.T, B), U)
	M = lamb - B

	init = union_vec(G, norm)

	#Computing largest eigenvector of M
	if eps == 0.:
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigsh(M, k=1, which='LA', v0=init)
			x = scipy.sparse.csr_matrix.dot(U, eigvecs[:,0])
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			try:
				(eigvals, eigvecs) = scipy.sparse.linalg.eigs(M, k=1, which='LR', v0=init)
				x = numpy.asarray(eigvecs[:,0])
				x = scipy.sparse.csr_matrix.dot(U, x)
			except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
				M = M.todense()
				(eigvals, eigvecs) = scipy.linalg.eigh(M, eigvals=((G.size() * G.num_snaps())-1,(G.size() * G.num_snaps())-1))
				x = eigvecs[:,0]
				x = scipy.sparse.csr_matrix.dot(U, x)
	else:
		x = power_method(M, init, eps)
		x = scipy.sparse.csr_matrix.dot(U, x)

	if norm is True:
		return sweep_norm(G, x.real)
	else:
		return sweep_sparse(G, x.real)

def independent_cut(G,norm=False):
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
		if norm is True:
			Lg = networkx.normalized_laplacian_matrix(G.snap(t), G.nodes())
		else:
			Lg = networkx.laplacian_matrix(G.snap(t), G.nodes())
		
		try:
			(eigvals, eigvecs) = scipy.sparse.linalg.eigs(Lg, k=2, which='SM')
			z = find_constant_eigenvector(eigvecs, eigvals)
		except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
			Ld = Lg.todense()
			(eigvals, eigvecs) = scipy.linalg.eigh(Ld, eigvals=(0, 1))
			z = find_constant_eigenvector(eigvecs, eigvals)
		
		if z == 0:
			x  = eigvecs[:,1].real
		else:
			x  = eigvecs[:,0].real
		
		if norm is True:
			c = sweep_norm_single(G.snap(t), x, G.nodes())
		else:
			c = sweep_sparse_single(G.snap(t), x, G.nodes())
		cs.append(c)
	
	#Aggregates the isoluated cuts into a temporal graph cut
	X = aggreg_opt_cuts(G, cs)

	edges_cut, swaps, den = evaluate_cut(G, X, norm)

	c = {}
	c["cut"] = X
	c["edges"] = edges_cut
	c["swaps"] = swaps
	c["score"] = (edges_cut + swaps) / den

	return c
		
def union_vec(G,norm=False):
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
				Gu[e[0]][e[1]]["weight"] = Gu[e[0]][e[1]]["weight"] + Gs[e[0]][e[1]]["weight"]
			else:
				Gu.add_edge(e[0],e[1],weight=Gs[e[0]][e[1]]["weight"])

	if norm is True:
		Lg = networkx.normalized_laplacian_matrix(Gu, G.nodes())
	else:
		Lg = networkx.laplacian_matrix(Gu, G.nodes())

	try:
		#Sometimes fails with an arpack error
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(Lg, k=2, which='SM')
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError, ValueError) as excpt:
		Ld = Lg.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Ld, eigvals=(0, 1))
	
	x  = eigvecs[:,numpy.argsort(eigvals)[1]].real
	
	C = numpy.array([])
	for t in range(G.num_snaps()):
		C = numpy.concatenate((C, x), axis=0)

	return C


def union_cut(G, norm=False):
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
				Gu[e[0]][e[1]]["weight"] = Gu[e[0]][e[1]]["weight"] + Gs[e[0]][e[1]]["weight"]
			else:
				Gu.add_edge(e[0],e[1],weight=Gs[e[0]][e[1]]["weight"])

	if norm is True:
		Lg = networkx.normalized_laplacian_matrix(Gu, G.nodes())
	else:
		Lg = networkx.laplacian_matrix(Gu, G.nodes())

	try:
		#Sometimes fails with an arpack error
		(eigvals, eigvecs) = scipy.sparse.linalg.eigs(Lg, k=2, which='SM')
	except (scipy.sparse.linalg.ArpackNoConvergence, scipy.sparse.linalg.ArpackError) as excpt:
		Ld = Lg.todense()
		(eigvals, eigvecs) = scipy.linalg.eigh(Ld, eigvals=(0, 1))
	
	x  = eigvecs[:,numpy.argsort(eigvals)[1]].real
	
	if norm is True:
		c = sweep_norm_single(Gu, x, G.nodes())
	else:
		c = sweep_sparse_single(Gu, x, G.nodes())
	
	#Extracting each cut
	X = numpy.array([])
	for t in range(G.num_snaps()):
		X = numpy.concatenate((X, c["cut"]), axis=0)
	
	edges_cut, swaps, den = evaluate_cut(G, X, norm)

	c = {}
	c["cut"] = X
	c["edges"] = edges_cut
	c["swaps"] = swaps
	c["score"] = (edges_cut + swaps) / den

	return c

def biased_cut(G, b):
	"""
		Computes optimal cut for snapshot t given vector b which captures information
		regarding previous and next snapshots. 
		Input:
			* G: temporal graph
			* b: bias
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

class TemporalCuts(object):
	"""
		Wrapper class for temporal cuts
	"""
	def __init__(self, name, algo, eps=0., k=1):
		self._name = name
		self.algos = ["indep-sparse", 
				"indep-norm", 
				"union-sparse", 
				"union-norm", 
				"inv-sparse",
				"inv-norm",
				"prod-sparse",
				"prod-norm",
				"diff-sparse",
				"diff-norm",
				"fast-sparse",
				"fast-norm",
				"laplacian-sparse",
				"laplacian-norm"]

		if algo not in self.algos:
			print("Unrecognized algorithm ", algo)

		self.algo = algo
		self.k = k
		self.eps = eps

	def name(self, name="name"):
		return self._name

	def set_rank(self, k):
		self.k = k	

	def cut(self, G):
		if self.algo == self.algos[0]:
			return independent_cut(G)
		elif self.algo == self.algos[1]:
			return independent_cut(G, norm=True)
		elif self.algo == self.algos[2]:
			return union_cut(G)
		elif self.algo == self.algos[3]:
			return union_cut(G, norm=True)
		elif self.algo == self.algos[4]:
			return inv_cut(G)
		elif self.algo == self.algos[5]:
			return inv_cut(G, norm=True)
		elif self.algo == self.algos[6]:
			return prod_cut(G)
		elif self.algo == self.algos[7]:
			return prod_cut(G, norm=True)
		elif self.algo == self.algos[8]:
			return diff_cut(G, norm=False, eps=self.eps)
		elif self.algo == self.algos[9]:
			return diff_cut(G, norm=True, eps=self.eps)
		elif self.algo == self.algos[10]:
			return fast_cut(G, self.k, norm=False, eps=self.eps)
		elif self.algo == self.algos[11]:
			return fast_cut(G, self.k, norm=True, eps=self.eps)
		elif self.algo == self.algos[12]:
			return laplacian_cut(G)
		elif self.algo == self.algos[13]:
			return laplacian_cut(G, norm=True)
		else:
			return None
