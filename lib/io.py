import networkx
import math
import numpy
import sys
from scipy import linalg
import random
import operator
import copy
from collections import deque
from datetime import datetime, date, time, timedelta
from lib.time_graph import *

def read_graph(input_graph_name, input_data_name):
	input_data = open(input_data_name, 'r')
	values = {}
    
	for line in input_data:
		line = line.rstrip()
		vec = line.rsplit(',')
        
		vertex = vec[0]
		value = float(vec[1])
		values[vertex] = value
        
	input_data.close()
    
	G = networkx.Graph()
    
	input_graph = open(input_graph_name, 'r')
    
	for line in input_graph:
		line = line.rstrip()
		vec = line.rsplit(',')
		v1 = vec[0]
		v2 = vec[1]

		if v1 in values and v2 in values:
			G.add_edge(v1,v2, weight=1.)
   
	Gcc=sorted(networkx.connected_component_subgraphs(G), key = len, reverse=True)

	G = Gcc[0]

	values_in_graph = {}

	for v in values.keys():
		if v in G:
			values_in_graph[v] = values[v]
	
	input_graph.close()
	networkx.set_node_attributes(G, values_in_graph, "value")
    
	return G

def read_time_graph(input_name, swp_cost=1., vertex_swp_cost=None, min_time=0, max_time=float('inf')):
	input_file = open(input_name, 'r')
	G = TimeGraph(swp_cost, vertex_swp_cost)

	for line in input_file:
		line = line.rstrip()
		vec = line.rsplit(',')
        
		v1 = vec[0]
		v2 = vec[1]
		t = int(vec[2])
		w = float(vec[3])
		
		if t <= max_time and t >= min_time:
			if v1 != v2:
				G.add_edge(v1, v2, t-min_time, w)

	input_file.close()

	return G

def read_values(input_data_name, G):
	D = {}
	input_data = open(input_data_name, 'r')

	for line in input_data:
		line = line.rstrip()
		vec = line.rsplit(',')

		vertex = vec[0]
		value = float(vec[1])
		D[vertex] = value

	input_data.close()
    
	F = []
	for v in G.nodes():
		if v in D:
			F.append(float(D[v]))
		else:
			F.append(0.)
   
	F = numpy.array(F)

	return F 

def read_time_values(path, G, snaps):
	FT = numpy.array([])
	for t in snaps:
		in_file = path + "_" + str(t) + ".data"
		F = read_values(in_file, G.snap(0))
		FT = numpy.concatenate((FT, F), axis=0)
        
	FT = FT - numpy.mean(FT)
	FT = FT / numpy.linalg.norm(FT)
    
	return FT 


