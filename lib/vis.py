import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
import scipy.fftpack
import random
import operator
import copy
from collections import deque
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
import os
import ast
import matplotlib.pyplot as plt

def set_f(G,F,ids=None):
	if ids is None:
		i = 0
		for v in G.nodes():
			G.node[v]["value"] = F[i]
			i = i + 1
	else:
		i = 0
		for v in range(len(ids)):
			G.node[ids[v]]["value"] = F[i]
			i = i + 1

def get_f(G):
	F = []
	for v in G.nodes():
		F.append(G.node[v]["value"])
	
	return numpy.array(F)

def rgb(minimum, maximum, value):
	if value > maximum:
		value = maximum

	if value < minimum:
		value = minimum

	mi, ma = float(minimum), float(maximum)
	ratio = 2 * (value-mi) / (ma - mi)
	b = int(max(0, 255*(1 - ratio)))
	r = int(max(0, 255*(ratio - 1)))
	g = 255 - b - r

	b = b / 255
	r = r / 255
	g = g / 255

	return (r, g, b)

def draw_graph_with_values(G, C, dot_output_file_name, maximum=None, minimum=None):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	if maximum is None:
		maximum = -sys.float_info.max
		minimum = sys.float_info.max

		for v in G.nodes():
			if G.node[v]["value"] > maximum:
				maximum = G.node[v]["value"]
	     
			if G.node[v]["value"] < minimum:
		 		minimum = G.node[v]["value"]
	i = 0
	for v in G.nodes():
		color = rgb(minimum, maximum, G.node[v]["value"])
		
		if C[i] > 0:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"1\",height=\"1\"];\n")
		else:
			output_file.write("\""+str(v)+"\" [shape=\"square\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"1\",height=\"1\"];\n")
		i = i + 1
	for edge in G.edges():
		output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")

	
	output_file.write("}")

	output_file.close()

def draw_graph_dynamic_values(G, FT, fig_output_file_name):
	maximum = -sys.float_info.max
	minimum = sys.float_info.max

	for i in range(FT.shape[0]):
		for j in range(FT.shape[1]):
			if FT[i][j] > maximum:
				maximum = FT[i][j]
			
			if FT[i][j] < minimum:
				minimum = FT[i][j]

	svg_names = ""

	for i in range(FT.shape[0]):
		set_f(G, FT[i])
		#draw_graph_with_values(G, "dyn_graph-"+str(i)+".dot", maximum, minimum)
		draw_graph_with_values(G, "dyn_graph-"+str(i)+".dot")
		os.system("sfdp -Goverlap=prism -Tsvg dyn_graph-"+str(i)+".dot > dyn_graph-"+str(i)+".svg")
		os.system("rm dyn_graph-"+str(i)+".dot")
		svg_names = svg_names + " dyn_graph-"+str(i)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

	for i in range(FT.shape[0]):
		os.system("rm dyn_graph-"+str(i)+".svg")
	
def draw_partitions_with_values(G, partitions, dot_output_file_name, maximum=None, minimum=None):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	if maximum is None:
		maximum = -sys.float_info.max
		minimum = sys.float_info.max

		for v in G.nodes():
			if G.node[v]["value"] > maximum:
				maximum = G.node[v]["value"]
	     
			if G.node[v]["value"] < minimum:
		 		minimum = G.node[v]["value"]
	
	part_map = {}

	for p in range(len(partitions)):
		for i in range(len(partitions[p])):
			part_map[partitions[p][i]] = p

	for v in G.nodes():
		color = rgb(minimum, maximum, G.node[v]["value"])
		if G.node[v]["value"] != 0.0:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"0.5\",height=\"0.5\"];\n")
		else:
			output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"0\",fixedsize=true,width=\"0.5\",height=\"0.5\"];\n")

	for edge in G.edges():
		if part_map[edge[0]] == part_map[edge[1]]:
			output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"4\"];\n")
		else:
			output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")
	
	output_file.write("}")

	output_file.close()

def draw_graph(G, dot_output_file_name):
	output_file = open(dot_output_file_name, 'w')
	output_file.write("graph G{\n")
	output_file.write("rankdir=\"LR\";\n")
	output_file.write("size=\"10,2\";\n")

	for v in G.nodes():
		color = rgb_to_hex(0,255,0)
		output_file.write("\""+str(v)+"\" [shape=\"circle\",label=\"\",style=filled,fillcolor=\""+str(color)+"\",penwidth=\"2\",fixedsize=true,width=\"1\",height=\"1\"];\n")

	for edge in G.edges():
		output_file.write("\""+str(edge[0])+"\" -- \""+str(edge[1])+"\"[dir=\"none\",color=\"black\",penwidth=\"1\"];\n")

	
	output_file.write("}")

	output_file.close()




def draw_time_graph_cut(G, fig_output_file_name, vec):
	svg_names = ""

	#Bulding union graph
	Gu = networkx.Graph()

	for v in G.snap(0).nodes():
		Gu.add_node(v)

	for t in range(G.num_snaps()):
		Gs = G.snap(t)
		for e in Gs.edges():
			#Weights are summed
			if Gs[e[0]][e[1]]["weight"] > G.connect_weight():
				if Gu.has_edge(e[0],e[1]):
					Gu[e[0]][e[1]]["weight"] = Gu[e[0]][e[1]]["weight"] + Gs[e[0]][e[1]]["weight"]
				else:
					Gu.add_edge(e[0],e[1],weight=Gs[e[0]][e[1]]["weight"])

	pos = networkx.fruchterman_reingold_layout(Gu) 
	
	for t in range(G.num_snaps()):
		plt.clf()
		posit = []
		neg = []
		others = []

		for v in G.snap(t).nodes():
			p = G.index_vertex(t, v)
			value = vec[p]
				
			if value < 0:
				neg.append(v)
			elif value > 0:
				posit.append(v)
			else:
				others.append(v)

		networkx.draw_networkx_nodes(G.snap(t),pos,node_size=120,nodelist=posit, node_color='red', node_shape='o')
		networkx.draw_networkx_nodes(G.snap(t),pos,node_size=120,nodelist=neg, node_color='blue', node_shape='v')

		if len(others) > 0:
			networkx.draw_networkx_nodes(G.snap(t),pos,node_size=100,nodelist=others, node_color='white', node_shape='s')
		
		edge_in_cluster = []
		edge_out_cluster = []
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]

			p1 = G.index_vertex(t, v1)
			p2 = G.index_vertex(t, v2)

			if vec[p1] == vec[p2]:
				edge_in_cluster.append(e)
			else:
				edge_out_cluster.append(e)
		
		networkx.draw_networkx_edges(G.snap(t), pos, width=0.5, edgelist=edge_in_cluster, edge_color='black', style='solid')
		networkx.draw_networkx_edges(G.snap(t), pos, width=0.5, edgelist=edge_out_cluster, edge_color='gray', style='dashed')
			
#networkx.draw_networkx_nodes(C,pos,node_size=100,node_list=G.nodes(), node_color=colors)
		#networkx.draw_networkx_edges(C, pos, width=.02, node_list=G.nodes())
		#networkx.draw_networkx_labels(C, pos, labels=labels, font_size=5, node_list=G.nodes())
#		networkx.draw_networkx_nodes(G.snap(t),pos,node_size=10,node_list=G.nodes(), node_color=colors)
#		networkx.draw_networkx_labels(G.snap(t), pos, labels=labels, font_size=5, node_list=G.nodes())
		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(wspace=-1,hspace=-1)
		plt.savefig("graph-"+str(t)+".svg", bbox_inches='tight', pad_inches=0) 
		svg_names = svg_names + " graph-"+str(t)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

#	for i in range(G.num_snaps()):
#		os.system("rm graph-"+str(i)+".svg")


def draw_time_graph_cut_values(G, cut, values, fig_output_file_name, pos=None, minimum=None, maximum=None):
	svg_names = ""
	
	new_pos = False
	if pos is None:
		#Bulding union graph
		Gu = networkx.Graph()
	
		for v in G.snap(0).nodes():
			Gu.add_node(v)
	
		for t in range(G.num_snaps()):
			Gs = G.snap(t)
			for e in Gs.edges():
				#Weights are summed
				if Gs[e[0]][e[1]]["weight"] > G.connect_weight():
					if Gu.has_edge(e[0],e[1]):
						Gu[e[0]][e[1]]["weight"] = Gu[e[0]][e[1]]["weight"] + Gs[e[0]][e[1]]["weight"]
					else:
						Gu.add_edge(e[0],e[1],weight=Gs[e[0]][e[1]]["weight"])

		pos = networkx.fruchterman_reingold_layout(Gu) 
		new_pos = True
	
	F = G.separate_values(values)

	if minimum is None:
		minimum = numpy.min(values)

	if maximum is None:
		maximum = numpy.max(values)

	C = G.separate_values(cut)
	
	for t in range(G.num_snaps()):
		plt.clf()
		f = F[t]
		c = C[t]
		
		labels = {}
		for v in G.nodes():
			labels[v] = v

		i = 0
		for v in G.nodes():
			color = rgb(minimum, maximum, f[i])
			if c[i] < 0:
				networkx.draw_networkx_nodes(G.snap(t),pos,node_size=400,nodelist=[v], node_color=[color], node_shape="o")
			elif c[i] > 0:
				networkx.draw_networkx_nodes(G.snap(t),pos,node_size=400,nodelist=[v], node_color=[color], node_shape="v")
			else:
				networkx.draw_networkx_nodes(G.snap(t),pos,node_size=400,nodelist=[v], node_color=[color], node_shape="s")

			i = i + 1

		edge_in_cluster = []
		edge_out_cluster = []
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]

			p1 = G.index_vertex(t, v1)
			p2 = G.index_vertex(t, v2)

			if cut[p1] == cut[p2]:
				edge_in_cluster.append(e)
			else:
				edge_out_cluster.append(e)
		
		networkx.draw_networkx_edges(G.snap(t), pos, width=1, edgelist=edge_in_cluster, edge_color='black', style='solid')
		networkx.draw_networkx_edges(G.snap(t), pos, width=1, edgelist=edge_out_cluster, edge_color='black', style='solid')
			
		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(wspace=-1,hspace=-1)
		plt.savefig("graph-"+str(t)+".svg", bbox_inches='tight', pad_inches=0) 
		svg_names = svg_names + " graph-"+str(t)+".svg"

	if new_pos is True:
		return pos
		

def draw_time_graph_partitions(G, fig_output_file_name, vec, show_labels=False):
	svg_names = ""

	#Bulding union graph
	Gu = networkx.Graph()

	for v in G.snap(0).nodes():
		Gu.add_node(v)

	for t in range(G.num_snaps()):
		Gs = G.snap(t)
		for e in Gs.edges():
			#Weights are summed
			if Gs[e[0]][e[1]]["weight"] > G.connect_weight():
				if Gu.has_edge(e[0],e[1]):
					Gu[e[0]][e[1]]["weight"] = Gu[e[0]][e[1]]["weight"] + Gs[e[0]][e[1]]["weight"]
				else:
					Gu.add_edge(e[0],e[1],weight=Gs[e[0]][e[1]]["weight"])

	pos = networkx.fruchterman_reingold_layout(Gu)
	
	if show_labels is True:
		max_degrees = {}
		degrees = Gu.degree()
		
		for t in range(G.num_snaps()):
			for v in Gu.nodes():
				p = G.index_vertex(t, v)
				part = vec[p] 

				if part not in max_degrees or degrees[max_degrees[part]] < degrees[v] :
					max_degrees[part] = v
		
	for t in range(G.num_snaps()):
		plt.clf()
		
		labels = {}
		if show_labels is True:
			for v in G.nodes():
				if v in max_degrees.values():
					labels[v] = v
				else:
					labels[v] = ""
		
		for c in range(0, 20):
			node_list = []
			
			for v in G.snap(t).nodes():
				p = G.index_vertex(t, v)
				value = vec[p]
				
				if value == c:
					node_list.append(v)

			if c == 0:
				color = 'skyblue'
				marker = 'o'
			elif c == 1:
				color = 'salmon'
				marker = 'v'
			elif c == 2:
				color = 'magenta'
				marker = 's'
			elif c == 3:
				color = 'lightgreen'
				marker = '*'
			elif c == 4:
				color = 'yellow'
				marker = 'D'
			elif c == 5:
				color = 'cyan'
				marker = '^'
			elif c == 6:
				color = 'magenta'
				marker = '<'
			else:
				color = 'white'
				marker = 's'

			if len(node_list) > 0:
				networkx.draw_networkx_nodes(G.snap(t),pos,node_size=100,nodelist=node_list, node_color=color, node_shape=marker, linewidths=0.)
		
		edge_in_cluster = []
		edge_out_cluster = []
		for e in G.snap(t).edges():
			v1 = e[0]
			v2 = e[1]

			p1 = G.index_vertex(t, v1)
			p2 = G.index_vertex(t, v2)

			if vec[p1] == vec[p2]:
				edge_in_cluster.append(e)
			else:
				edge_out_cluster.append(e)
		
		networkx.draw_networkx_labels(G.snap(t), pos, labels=labels, font_size=15, node_list=G.nodes())
		networkx.draw_networkx_edges(G.snap(t), pos, width=1, edgelist=edge_in_cluster, edge_color='black', style='solid')
		networkx.draw_networkx_edges(G.snap(t), pos, width=1, edgelist=edge_out_cluster, edge_color='gray', style='dashed')
			
#		networkx.draw_networkx_nodes(C,pos,node_size=100,node_list=G.nodes(), node_color=colors)
		#networkx.draw_networkx_edges(C, pos, width=.02, node_list=G.nodes())
		#networkx.draw_networkx_labels(C, pos, labels=labels, font_size=5, node_list=G.nodes())
#		networkx.draw_networkx_nodes(G.snap(t),pos,node_size=10,node_list=G.nodes(), node_color=colors)
#		networkx.draw_networkx_labels(G.snap(t), pos, labels=labels, font_size=5, node_list=G.nodes())
		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(wspace=-1,hspace=-1)
		plt.savefig("graph-"+str(t)+".svg", bbox_inches='tight', pad_inches=0) 
		svg_names = svg_names + " graph-"+str(t)+".svg"
		
	os.system("python lib/svg_stack-master/svg_stack.py --direction=v --margin=0 "+svg_names+" > "+fig_output_file_name)

def draw_time_graph_grid(G, values, fig_output_file_name):
	svg_names = ""
	svg_names = ""

	#Bulding union graph
	Gu = networkx.Graph()

	for t in range(G.num_snaps()):
		plt.clf()

		maximum = -sys.float_info.max
		minimum = sys.float_info.max
		
		for v in G.nodes():
			if values[t][v] < minimum:
				minimum = values[t][v]

			if values[t][v] > maximum:
				maximum = values[t][v]
		pos = {}

		for v in G.nodes():
			pos[v] = v

		i = 0
		for v in G.nodes():
			color = rgb(minimum, maximum, values[t][v])

			networkx.draw_networkx_nodes(G.snap(t),pos,node_size=100,nodelist=[v], node_color=color, node_shape="o")

		networkx.draw_networkx_edges(G.snap(t), pos, width=1, edgelist=G.snap(t).edges(), edge_color='black', style='solid')
			
		plt.axis('off')
		plt.tight_layout()
		plt.subplots_adjust(wspace=-1,hspace=-1)
		plt.savefig("graph-"+str(t)+".svg", bbox_inches='tight', pad_inches=0) 
		svg_names = svg_names + " graph-"+str(t)+".svg"
		
