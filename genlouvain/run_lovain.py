import os

def run_lovain_method(input_file_name, omega):
	output_file_name = "out_lovain.csv"
	os.system("/usr/local/MATLAB/R2016b/bin/matlab -nodisplay -nodesktop -r \"run_lovain(\'"+input_file_name+"\',\'"+output_file_name+"\',"+str(omega)+"); exit;\"")
	print("/usr/local/MATLAB/R2016b/bin/matlab -nodisplay -nodesktop -r \'run_lovain("+input_file_name+","+output_file_name+","+str(omega)+"); exit;\'")
        
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
