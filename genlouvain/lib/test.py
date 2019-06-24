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
from matplotlib.lines import Line2D
from lib.io import *
from lib.vis import *
from lib.graph_signal_proc import *
from lib.netpros import *
from lib.syn import *
from lib.experiments import *
from lib.static import *
from lib.datasets import *

G = read_graph(traffic["path"] + "traffic.graph", traffic["path"] + "traffic_100.data")
F = read_dyn_graph(traffic["path"] + "traffic", traffic["num_snaps"], G) 


FT_h = []

for i in range(0, 12*5, 12):
	FT_h.append(F[i])

FT_h = numpy.array(FT_h)

algs = [OptWavelets(n=5)]

comp_ratios = [0.3]

res, time = compression_experiment_static(G, FT_h[1:2], algs, comp_ratios)
