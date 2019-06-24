import networkx
import math
import scipy.optimize
import numpy
import sys
from scipy import linalg
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

class Fourier(object):    
	def name(self):
		return "FT"

	def set_graph(self, _G):
		self.G = _G
		L = networkx.laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)

	def transform(self, F):
		"""
		"""
		return graph_fourier(F, self.U)

	def inverse(self, ftr):
		"""
		"""
		return graph_fourier_inverse(ftr, self.U)

	def drop_frequency(self, ftr, n):
		coeffs = {}

		for i in range(ftr.shape[0]):
			coeffs[i] = abs(ftr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		ftr_copy = numpy.copy(ftr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			ftr_copy[i] = 0

		return ftr_copy

class HWavelets(object):
	def name(self):
		return "HWT"

	def set_graph(self, _G):
		self.G = _G
		L = networkx.normalized_laplacian_matrix(self.G)
		L = L.todense()
		self.U, self.lamb_str = compute_eigenvectors_and_eigenvalues(L)
		lamb_max = max(self.lamb_str.real)
		K = 100
		J = 4
		gamma = comp_gamma()
		self.T = comp_scales(lamb_max, K, J)
		self.w = graph_wavelets(self.lamb_str.real, self.U.real, range(len(self.G.nodes())), self.T)
		self.s = graph_low_pass(self.lamb_str.real, self.U.real, range(len(self.G.nodes())), self.T, gamma, lamb_max, K)

	def transform(self, F):
		"""
		"""
		return hammond_wavelet_transform(self.w, self.s, self.T, F)
	
	def inverse(self, wtr):
		"""
		"""
		return hammond_wavelets_inverse(self.w, self.s, wtr)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			for j in range(len(wtr[i])):
				coeffs[(i,j)] = abs(wtr[i][j])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0][0]
			j = sorted_coeffs[k][0][1]

			wtr_copy[i][j] = 0.0

		return wtr_copy

class GWavelets(object):
	def name(self):
		return "Gavish-Random"

	def set_graph(self, _G):
		self.G = _G
		(self.tree, self.ind) = gavish_hierarchy(self.G, 1)

	def transform(self, F):
		"""
		"""
		return gavish_wavelet_transform(self.tree, self.ind, self.G, F)

	def inverse(self, wtr):
		"""
		"""
		return gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			wtr_copy[i] = 0.0

		return wtr_copy

class GNCWavelets(object):
	def name(self):
		return "GWT"

	def set_graph(self, _G):
		self.G = _G
		(self.tree, self.ind) = normalized_cut_hierarchy(self.G)

	def transform(self, F):
		"""
		"""
		return gavish_wavelet_transform(self.tree, self.ind, self.G, F)

	def inverse(self, wtr):
		"""
		"""
		return gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			wtr_copy[i] = 0.0

		return wtr_copy


class PWavelets(object):
	def __init__(self, k):
		self.k = k
	
	def name(self):
		return "Hierarchical-Laplacian"

	def set_graph(self, _G):
		self.G = _G
	
	def transform(self, F):
		"""
		"""
		set_weight_graph(self.G, F)
		(self.tree, self.ind) = pyramid_hierarchy(self.G, self.k)
		return pyramid_wavelet_transform(self.tree, self.ind, self.G, F)

	def inverse(self, wtr):
		"""
		"""
		return pyramid_wavelet_inverse(self.tree, self.ind, self.G, wtr)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		for i in range(len(wtr)):
			coeffs[i] = abs(wtr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(wtr)

		for k in range(n, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			wtr_copy[i] = 0.0

		return wtr_copy

class OptWavelets(object):
	def __init__(self, n=0):
		self.n = n
	
	def name(self):
		if self.n == 0:
			return "SWT"
		else:
			return "FSWT"

	def set_graph(self, _G):
		self.G = _G

	def transform(self, F):
		"""
		"""
		self.F = F
		return None

	def inverse(self, wtr):
		"""
		"""
		return gavish_wavelet_inverse(self.tree, self.ind, self.G, wtr)

	def drop_frequency(self, wtr, n):
		coeffs = {}
		
		k = n
		(self.tree, self.ind, s) =  optimal_wavelet_basis(self.G, self.F, k, self.n)
		tr = gavish_wavelet_transform(self.tree, self.ind, self.G, self.F)
		
		for i in range(len(tr)):
			coeffs[i] = abs(tr[i])

		sorted_coeffs = sorted(coeffs.items(), key=operator.itemgetter(1), reverse=True)

		wtr_copy = numpy.copy(tr)
		v = n - int(math.ceil(float(s * math.log2(len(self.G.edges()))) / 64))
		print(n ," x ", v)
		for k in range(v, len(sorted_coeffs)):
			i = sorted_coeffs[k][0]

			wtr_copy[i] = 0.0

		return wtr_copy

