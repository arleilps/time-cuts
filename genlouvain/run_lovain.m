
function run_lovain(input_file_name,out_file_name,omega)
[A,V] = read_graph(input_file_name);
S = gen_lovain_wrapper(A,V,omega);
write_lovain_output(S,V,out_file_name);