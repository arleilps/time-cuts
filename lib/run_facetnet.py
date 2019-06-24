import os

def run_facetnet_method(input_file_name, K, lamb):
	output_file_name = "out_facetnet.csv"
	os.system("/usr/local/MATLAB/R2012a/bin/matlab -nodisplay -nodesktop -r \"run_facetnet(\'"+input_file_name+"\',\'"+output_file_name+"\',"+str(K)+","+str(lamb)+"); exit;\"")
	print("/usr/local/MATLAB/R2012a/bin/matlab -nodisplay -nodesktop -r \"run_facetnet(\'"+input_file_name+"\',\'"+output_file_name+"\',"+str(K)+","+str(lamb)+"); exit;\"")
        
	out_file = open(output_file_name, 'r')
	P = {}
        
	for line in out_file:
		line = line.rstrip()
		vec = line.rsplit(',')
		
		v = vec[0]
		t = int(vec[1])
		c = int(vec[2])
		
		if v not in P:
			P[v] = {}

		P[v][t] = c

	return P
