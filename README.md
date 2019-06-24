# Implementation of spectral algorithms for temporal graph cuts.

Evaluation is performed using python notebooks.

Performance experiments:

Compression experiments:

The main functions are implemented in time_graph.py, which includes a wrapper for several temporal cut algorithms.

Example of usage:

#Reads input graph (primary school) with swap cost of .0249

G = read_time_graph(primary_school["graph"], .0249)

G.make_connected(0.005)

c = prod_cut(G)

print(c)

({'cut': array([-1., -1.,.. -1.]),
  'score': 9.722578317913037e-05,
  'edges': 3.991778580973271,
  'swaps': 0.04979999999999988},
 array([-0.02384887, -0.02996254,... -0.06425752]))


Notice that we purposefully did not include the code for the baseline community detection methods, 
but some wrappers that can be used  to reproduce the comparison. The source-code for the baselines 
can be found in the following links:

FacetNet: https://ame2.asu.edu/students/lin/code/snmf_evol.zip
GenLovain: http://netwiki.amath.unc.edu/GenLouvain/GenLouvain 

For more details, see the paper:  
[Spectral Algorithms for Temporal Graph Cuts ](http://www.cs.ucsb.edu/~arlei/pubs/www18.pdf "")  
Arlei Silva, Ambuj K Singh, Ananthram Swami  
The Web Conference (WWW), 2018. 

Arlei Silva (arlei@cs.ucsb.edu)
