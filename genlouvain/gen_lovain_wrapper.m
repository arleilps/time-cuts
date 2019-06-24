function [S] = gen_lovain_wrapper(A,V,omega)
% Run the genLovain method on matrix A
% Input:
%    + A: Array of cells with adjacency matrices of snapshots
%    + omega: Coupling parameter
% Output:
%    + S: community assingments
gamma=1.
N=size(V,1);
T=length(A);
B=spalloc(N*T,N*T,N*N*T+2*N*T);
twomu=0;
for s=1:T
    k=sum(A{s});
    twom=sum(k);
    twomu=twomu+twom;
    indx=[1:N]+(s-1)*N;
    B(indx,indx)=A{s}-gamma*k'*k/twom;
end
twomu=twomu+2*omega*N*(T-1);
B = B + omega*spdiags(ones(N*T,2),[-N,N],N*T,N*T);
[S,Q] = genlouvain(B);
Q = Q/twomu
S = reshape(S,N,T);
