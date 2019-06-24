function run_facetnet(input_file_name, out_file_name, K, lamb)
[A,V] = read_graph(input_file_name);

fid = fopen(out_file_name,'wt');

X = A{1}
[D, H, logl] = snmf_evol(X, K, 1e-6, 500);
Y=H./repmat(sum(H,2),[1 2]);

for v = 1:size(V,1)
	[val ind] = max(Y(v,:));
	fprintf(fid, '%d,0,%d\n', V(v), ind-1);
end

for i = 2:size(A,1)
	S{i} = zeros(size(V,1))
	X = A{i}
	[D, H, logl] = snmf_evol(X, K, 1e-6, 500,D, H, lamb, H);
	Y=H./repmat(sum(H,2),[1 2]);

	for v = 1:size(V,1)
		[val ind] = max(Y(v,:));
		fprintf(fid, '%d,%d,%d\n', V(v),i-1,ind-1);
	end
end

fclose(fid);
