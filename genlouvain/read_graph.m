function [DynGM V] = read_graph(filename)
% Reads dynamic graph from .csv
% Input: 
%	+ filename: format v1,v2,weight,timestamp
% Output:
%	+ DynGM: cell with one adjacency matrix for each snapshot
%       + V: list of vertex ids
% [dyn,V] = read_graph('../../data/primary_school/graph.csv');
M = csvread(filename);
n_snaps = size(unique(M(:,3)),1);
DynGM = cell(n_snaps,1);
V = unique([M(:,1); M(:,2)]);
num_v = size(V,1);
for i = 1:n_snaps
    DynGM{i} = sparse(num_v,num_v);
end
m = containers.Map('KeyType','int32','ValueType','int32')
ID=1
for i = 1:size(V,1)
    m(V(i))=ID;
    ID = ID + 1;
end
for i = 1:size(M,1)
    snp = M(i,3)+1;
    w = M(i,4);
    v1 = m(M(i,1));
    v2 = m(M(i,2));
    DynGM{snp}(v1,v2) = w;
    DynGM{snp}(v2,v1) = w;
end
