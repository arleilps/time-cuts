function write_lovain_output(S,V,out_file_name)
% Writes community assignments computed by genLovain
% Input:
%    + S: community assignments
%    + V: vertex identifiers
out_file = fopen(out_file_name,'w');
for i = 1:size(S,1)
    for j = 1:size(S,2)
        v=V(i);
        c=S(i,j)-1;
        t=j-1;
        fprintf(out_file, '%d,%d,%d\n',v,t,c);
    end
end
fclose(out_file)
        
