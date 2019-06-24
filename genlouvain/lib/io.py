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
import pandas as pd
from datetime import datetime, date, time, timedelta
import statsmodels.api as sm
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
	networkx.set_node_attributes(G, "value", values_in_graph)
    
	return G

def read_time_graph(input_name, swp_cost=1., min_time=0, max_time=float('inf')):
	input_file = open(input_name, 'r')
	G = TimeGraph(swp_cost)

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
	F = F / numpy.max(F)
	F = F - numpy.mean(F)
    
	return F 

def read_dyn_graph(path, num_snapshots, G):
	FT = []
	for t in range(num_snapshots):
		in_file = path + "_" + str(t) + ".data"
		F = read_values(in_file, G)
		FT.append(F)
        
	return numpy.array(FT)

def clean_traffic_data(FT):
	start_time = datetime.strptime("1/04/11 00:00", "%d/%m/%y %H:%M")
	c_FT = []
	for i in range(FT.shape[1]):
		#removing daily seasonality
		data = pd.DataFrame(FT[:,i], pd.DatetimeIndex(start='1/04/11 00:00', periods=len(FT[:,i]), freq='5min'))
		data.interpolate(inplace=True)
		
		res = sm.tsa.seasonal_decompose(data.values, freq=288)
		F = FT[:,i] - res.seasonal

		#removing weekly seasonality
		data = pd.DataFrame(F, pd.DatetimeIndex(start='1/04/11 00:00', periods=len(FT[:,i]), freq='5min'))
		res = sm.tsa.seasonal_decompose(data.values, freq=288*7)
		F = F - res.seasonal
		
		c_FT.append(F)

	return numpy.array(c_FT).transpose()


