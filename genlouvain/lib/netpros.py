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

from lib.graph_signal_proc import *
from lib.optimal_cut import *

def build_stacked_graph_dense(G, FT):
	GT = networkx.Graph()
	values = {}

	for e in G.edges():
		v1 = str(e[0])+"-0"
		v2 = str(e[1])+"-0"
		GT.add_edge(v1, v2)

	for t in range(1, FT.shape[0]):
		for e in G.edges():
			v1 = str(e[0])+"-"+str(t)
			v2 = str(e[1])+"-"+str(t)
			GT.add_edge(v1, v2)

	for t in range(0, FT.shape[0]-1):
		for v in G.nodes():
			GT.add_edge(str(v)+"-"+str(t), str(v)+"-"+str(t+1))
			for n in G.neighbors(v):
				GT.add_edge(str(v)+"-"+str(t), str(n)+"-"+str(t+1))

	F = []
	for t in range(0, FT.shape[0]):
		i = 0
		for v in G.nodes():
			GT.node[str(v)+"-"+str(t)]["value"] = FT[t][i]
			i = i + 1
		
	for v in GT.nodes():
		F.append(GT.node[v]["value"])
	
	return GT, numpy.array(F)

def values_stacked_graph(F, G, G_unstacked):
	FT = []
	snps = int(len(F)/len(G_unstacked.nodes()))

	values = {}
	i = 0
	for v in G.nodes():
		values[v] = F[i]
		i = i + 1

	for t in range(snps):
		FT.append([])
		for v in G_unstacked.nodes():
			FT[t].append(values[str(v)+"-"+str(t)])

	return numpy.array(FT)

class TwoDFourier(object):    
	def name(self):
		return "2D-Fourier"

	def set_graph(self, _G):
		self.G = _G
		L = networkx.normalized_laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)

	def transform(self, F):
		"""
		"""
		f1 = []
		for a in range(F.shape[0]):
			gf = graph_fourier(F[a], self.U)
			f1.append(gf)

		f1 = numpy.array(f1)
		f2 = []

		for u in range(f1.shape[1]):
			ft = fourier_transform(f1[:,u])
			f2.append(ft)

		f2 = numpy.array(f2)

		return f2

	def inverse(self, ftr):
		"""
		"""
		f1 = []
		for a in range(ftr.shape[0]):
			ft = fourier_inverse(ftr[a])
			f1.append(ft)

		f1 = numpy.array(f1).transpose()
		F = []    

		for v in range(f1.shape[0]):
			gf = graph_fourier_inverse(f1[v], self.U)
			F.append(gf)

		return numpy.array(F)

	def drop_frequency(self, ftr, n):
		coeffs = {}

		for i in range(ftr.shape[0]):
			for j in range(ftr.shape[1]):
				coeffs[(i,j)] = abs(ftr[i][j])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		ftr_copy = numpy.copy(ftr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]

			ftr_copy[i][j] = 0

		return ftr_copy

	def scale_energy_info(self, ftr):
		sei = []
		s = 0
		for i in range(ftr.shape[0]):
			sei.append([])
		for j in range(ftr.shape[1]):
			sei[i].append(abs(ftr[i][j]))
			s = s + abs(ftr[i][j])

		return numpy.array(sei) / s

class OneDFourier(object):
	def name(self):
		return "1D-Fourier"

	def set_graph(self, _G):
		self.G_unstacked = _G

	def transform(self, _F):
		"""
		"""
		(self.G, self.F) = build_stacked_graph_dense(self.G_unstacked, _F)
		L = networkx.normalized_laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)

		return graph_fourier(self.F, self.U)

	def inverse(self, ftr):
		"""
		"""
		inv_stacked = graph_fourier_inverse(ftr, self.U)

		return values_stacked_graph(inv_stacked, self.G, self.G_unstacked)

	def drop_frequency(self, ftr, n):
		"""
		"""
		coeffs = {}

		for i in range(len(ftr)):
			coeffs[i] = abs(ftr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		ftr_copy = numpy.copy(ftr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]
			ftr_copy[i] = 0

		return ftr_copy  

def L2(F, F_approx):
	e = 0
	for i in range(F.shape[0]):
		e = e + ((F[i]-F_approx[i])**2).sum()

	return float(e) 

def L1(F, F_approx):
	e = 0
	for i in range(F.shape[0]):
		e = e + (abs(F[i]-F_approx[i])).sum()

	return float(e)

class TwoDHWavelets(object):
	def name(self):
		return "2D-Hammond"

	def set_graph(self, _G):
		self.G = _G
		L = networkx.normalized_laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)
		lamb_max = max(self.lamb_str.real)
		K = 10
		J = 10
		gamma = comp_gamma()
		self.T = comp_scales(lamb_max, K, J)
		self.w = graph_wavelets(self.lamb_str.real, self.U.real, range(len(self.G.nodes())), self.T)
		self.s = graph_low_pass(self.lamb_str.real, self.U.real, range(len(self.G.nodes())), self.T, gamma, lamb_max, K)

	def transform(self, F):
		"""
		"""
		f1 = []
		for a in range(F.shape[0]):
			wv = hammond_wavelet_transform(self.w, self.s, self.T, F[a])
			f1.append(wv)

		db1 = pywt.Wavelet('db1')

		f1 = numpy.array(f1)
		f2 = []

		for a in range(f1.shape[1]):
			f2.append([])
			for v in range(f1.shape[2]):
				wv = pywt.wavedec(f1[:,a,v].real, db1)
				f2[a].append(wv)

		return f2

	def inverse(self, wtr):
		"""
		"""
		f1 = []
		for a in range(len(wtr)):
			f1.append([])
			for v in range(len(wtr[a])):
				wv = pywt.waverec(wtr[a][v], 'db1')
				f1[a].append(wv)
		F = []    
		f1 = numpy.array(f1).transpose((2,0,1))

		for t in range(f1.shape[0]):
			f = hammond_wavelets_inverse(self.w, self.s, f1[t])
			F.append(f)

		return numpy.array(F)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				for m in range(len(wtr[i][j])):
					for p in range(len(wtr[i][j][m])):
						coeffs[(i,j,m,p)] = abs(wtr[i][j][m][p])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = copy.deepcopy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]
			m = sorted_coeffs[k][0][2]
			p = sorted_coeffs[k][0][3]

			wtr_copy[i][j][m][p] = 0.0

		return wtr_copy

	def scale_energy_info(self, wtr):
		sei = []
		s = 0

		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				sei.append([])
				for m in range(len(wtr[i][j])):
					for p in range(len(wtr[i][j][m])):
						sei[-1].append(abs(wtr[i][j][m][p]))
						s = s + abs(wtr[i][j][m][p])

		return numpy.array(sei) / s

class OneDHWavelets(object):
	def name(self):
		return "1D-Hammond"

	def set_graph(self, _G):
		self.G_unstacked = _G

	def transform(self, _F):
		"""
		"""
		(self.G, self.F) = build_stacked_graph_dense(self.G_unstacked, _F)
		L = networkx.normalized_laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)
		lamb_max = max(self.lamb_str)
		gamma = comp_gamma()
		K = 10
		J = 10
		self.T = comp_scales(lamb_max, K, J)
		self.w = graph_wavelets(self.lamb_str, self.U, range(len(self.G.nodes())), self.T)
		self.s = graph_low_pass(self.lamb_str, self.U, range(len(self.G.nodes())), self.T, gamma, lamb_max, K)

		return hammond_wavelet_transform(self.w, self.s, self.T, self.F)

	def inverse(self, wtr):
		"""
		"""
		inv_stacked =  hammond_wavelets_inverse(self.w, self.s, wtr)

		return values_stacked_graph(inv_stacked, self.G, self.G_unstacked)

	def drop_frequency(self, wtr, n):
		coeffs = {}

		for i in range(wtr.shape[0]):
			for j in range(wtr.shape[1]):
				coeffs[(i,j)] = abs(wtr[i][j])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]

			wtr_copy[i][j] = 0

		return wtr_copy

class TwoDGWavelets(object):
	def name(self):
		return "2D-Gavish-Random"

	def set_graph(self, _G):
		self.G = _G
		(self.tree, self.ind) = gavish_hierarchy(self.G, 1)

	def transform(self, F):
		"""
		"""
		f1 = []
		for a in range(F.shape[0]):
			wv = gavish_wavelet_transform(self.tree, self.ind, self.G, F[a])
			f1.append(wv)

		db1 = pywt.Wavelet('db1')

		f1 = numpy.array(f1)
		f2 = []

		for a in range(f1.shape[1]):
			wv = pywt.wavedec(f1[:,a], db1)
			f2.append(wv)

		return f2

	def inverse(self, wtr):
		"""
		"""
		f1 = []
		for a in range(len(wtr)):
			wv = pywt.waverec(wtr[a], 'db1')
			f1.append(wv)

		F = []    
		f1 = numpy.array(f1).transpose()

		for t in range(f1.shape[0]):
			f = gavish_wavelet_inverse(self.tree, self.ind, self.G, f1[t])
			F.append(f)

		return numpy.array(F)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					coeffs[(i,j,k)] = abs(wtr[i][j][k])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = copy.deepcopy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]
			k = sorted_coeffs[k][0][2]

			wtr_copy[i][j][k] = 0.0

		return wtr_copy

	def scale_energy_info(self, wtr):
		sei = []
		s = 0
		for i in range(len(wtr)):
			sei.append([])
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					sei[i].append(abs(wtr[i][j][k]))
					s = s + abs(wtr[i][j][k])

		return numpy.array(sei) / s

class OneDGWavelets(object):
	def name(self):
		return "1D-Gavish-Random"

	def set_graph(self, _G):
		self.G_unstacked = _G

	def transform(self, _F):
		"""
		"""
		(self.G, self.F) = build_stacked_graph_dense(self.G_unstacked, _F)
		(self.tree, self.ind) = gavish_hierarchy(self.G, 1)

		return gavish_wavelet_transform(self.tree, self.ind, self.G, self.F)

	def inverse(self, wtr):
		"""
		"""
		inv_stacked =  gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

		return values_stacked_graph(inv_stacked, self.G, self.G_unstacked)

	def drop_frequency(self, wtr, n):
		coeffs = {}

		for i in range(wtr.shape[0]):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]
			wtr_copy[i] = 0

		return wtr_copy



class TwoDGNCWavelets(object):
	def name(self):
		return "2D-Gavish-Norm-Cut"

	def set_graph(self, _G):
		self.G = _G
		(self.tree, self.ind) = normalized_cut_hierarchy(self.G)

	def transform(self, F):
		"""
		"""
		f1 = []
		for a in range(F.shape[0]):
			wv = gavish_wavelet_transform(self.tree, self.ind, self.G, F[a])
			f1.append(wv)

		db1 = pywt.Wavelet('db1')

		f1 = numpy.array(f1)
		f2 = []

		for a in range(f1.shape[1]):
			wv = pywt.wavedec(f1[:,a], db1)
			f2.append(wv)

		return f2

	def inverse(self, wtr):
		"""
		"""
		f1 = []
		for a in range(len(wtr)):
			wv = pywt.waverec(wtr[a], 'db1')
			f1.append(wv)

		F = []    
		f1 = numpy.array(f1).transpose()

		for t in range(f1.shape[0]):
			f = gavish_wavelet_inverse(self.tree, self.ind, self.G, f1[t])
			F.append(f)

		return numpy.array(F)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					coeffs[(i,j,k)] = abs(wtr[i][j][k])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = copy.deepcopy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]
			k = sorted_coeffs[k][0][2]

			wtr_copy[i][j][k] = 0.0

		return wtr_copy

	def scale_energy_info(self, wtr):
		sei = []
		s = 0
		for i in range(len(wtr)):
			sei.append([])
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					sei[i].append(abs(wtr[i][j][k]))
					s = s + abs(wtr[i][j][k])

		return numpy.array(sei) / s

class OneDGNCWavelets(object):
	def name(self):
		return "1D-Gavish-Norm-Cut"

	def set_graph(self, _G):
		self.G_unstacked = _G

	def transform(self, _F):
		"""
		"""
		(self.G, self.F) = build_stacked_graph_dense(self.G_unstacked, _F)
		(self.tree, self.ind) = normalized_cut_hierarchy(self.G)

		return gavish_wavelet_transform(self.tree, self.ind, self.G, self.F)

	def inverse(self, wtr):
		"""
		"""
		inv_stacked =  gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

		return values_stacked_graph(inv_stacked, self.G, self.G_unstacked)

	def drop_frequency(self, wtr, n):
		coeffs = {}

		for i in range(wtr.shape[0]):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			wtr_copy[i] = 0

		return wtr_copy

class SVD(object):
	def name(self):
		return "SVD"

	def set_graph(self, _G):
		self.G = _G

	def transform(self, _F):
		"""
		"""
		return svd_transform(_F)

	def inverse(self, svd):
		"""
		"""
		return svd_inverse(svd[0], svd[1], svd[2])

	def drop_frequency(self, svd, n):
		"""
		"""
		n_svd = int(math.floor(float(n) / (svd[0].shape[1] + svd[2].shape[0])))

		return svd[0], svd_filter(svd[0], numpy.copy(svd[1]), svd[2], n_svd), svd[2]    

class TwoDPWavelets(object):
	def __init__(self):
		self.k = 10
	
	def name(self):
		return "2D-Pyramid"

	def set_graph(self, _G):
		self.G = _G
		(self.tree, self.ind) = pyramid_hierarchy(self.G, k)
	
	def set_k(self, k):
		self.k = k

	def transform(self, F):
		"""
		"""
		f1 = []
		for a in range(F.shape[0]):
			wv = pyramid_wavelet_transform(self.tree, self.ind, self.G, F[a])
			f1.append(wv)

		db1 = pywt.Wavelet('db1')

		f1 = numpy.array(f1)
		f2 = []

		for a in range(f1.shape[1]):
			wv = pywt.wavedec(f1[:,a], db1)
			f2.append(wv)

		return f2

	def inverse(self, wtr):
		"""
		"""
		f1 = []
		for a in range(len(wtr)):
			wv = pywt.waverec(wtr[a], 'db1')
			f1.append(wv)

		F = []    
		f1 = numpy.array(f1).transpose()

		for t in range(f1.shape[0]):
			f = pyramid_wavelet_inverse(self.tree, self.ind, self.G, f1[t])
			F.append(f)

		return numpy.array(F)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					coeffs[(i,j,k)] = abs(wtr[i][j][k])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = copy.deepcopy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]
			k = sorted_coeffs[k][0][2]

			wtr_copy[i][j][k] = 0.0

		return wtr_copy


	def scale_energy_info(self, wtr):
		sei = []
		s = 0
		for i in range(len(wtr)):
			sei.append([])
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					sei[i].append(abs(wtr[i][j][k]))
					s = s + abs(wtr[i][j][k])

		return numpy.array(sei) / s

class OneDPWavelets(object):
	def __init__(self):
		self.k = 10
	
	def name(self):
		return "1D-Pyramid"

	def set_graph(self, _G):
		self.G_unstacked = _G
	
	def set_k(self, k):
		self.k = k

	def transform(self, _F):
		"""
		"""
		(self.G, self.F) = build_stacked_graph_dense(self.G_unstacked, _F)
		(self.tree, self.ind) = pyramid_hierarchy(self.G, k)

		return pyramid_wavelet_transform(self.tree, self.ind, self.G, self.F)

	def inverse(self, wtr):
		"""
		"""
		inv_stacked =  pyramid_wavelet_inverse(self.tree, self.ind, self.G, wtr)

		return values_stacked_graph(inv_stacked, self.G, self.G_unstacked)

	def drop_frequency(self, wtr, n):
		coeffs = {}

		for i in range(wtr.shape[0]):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]
			wtr_copy[i] = 0

		return wtr_copy

class TwoDOptWavelets(object):
	def __init__(self, k, beta, G, F, n):
		self.k = k
		self.beta = beta
		(self.tree, self.ind) = best_tree_avg(G, F, k, beta, n)	

	def name(self):
		return "2D-Sparse-cut"

	def set_graph(self, _G):
		self.G = _G

	def transform(self, F):
		"""
		"""
		f1 = []
		T = []

		for a in range(F.shape[0]):
			wv = gavish_wavelet_transform(self.tree, self.ind, self.G, F[a])
			f1.append(wv)

		db1 = pywt.Wavelet('db1')

		f1 = numpy.array(f1)
		f2 = []

		for a in range(f1.shape[1]):
			wv = pywt.wavedec(f1[:,a], db1)
			f2.append(wv)

		return f2

	def inverse(self, wtr):
		"""
		"""
		f1 = []
		for a in range(len(wtr)):
			wv = pywt.waverec(wtr[a], 'db1')
			f1.append(wv)

		F = []    
		f1 = numpy.array(f1).transpose()

		for t in range(f1.shape[0]):
			f = gavish_wavelet_inverse(self.tree, self.ind, self.G, f1[t])
			F.append(f)

		return numpy.array(F)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				for k in range(len(wtr[i][j])):
					coeffs[(i,j,k)] = abs(wtr[i][j][k])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = copy.deepcopy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]
			k = sorted_coeffs[k][0][2]

			wtr_copy[i][j][k] = 0.0

		return wtr_copy

