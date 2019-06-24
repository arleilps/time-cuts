from Estrangement import estrangement
from Estrangement import plots
from Estrangement import options_parser

def read_estrangement_output(out_file_name):
	P = eval(open(out_file_name, 'r').read())
	S = {}
	comms = {}
	ID = 0
	for t in P:
		for v in P[t]:
			if P[t][v] not in comms:
				comms[P[t][v]] = ID
				ID = ID + 1
			
			if str(v) not in S:
				S[str(v)] = {}

			S[str(v)][t-1] = comms[P[t][v]]

	for v in S:	
		for t in P:
			if t-1 not in S[v]:
				S[v][t-1] = ID
				ID = ID + 1

	return S	

def run_estrangement(dataset, delt):
	out_file_name = "results.dat"
	estrangement.ECA(dataset_dir=dataset, delta=delt, results_filename=out_file_name)
	return read_estrangement_output(out_file_name)
