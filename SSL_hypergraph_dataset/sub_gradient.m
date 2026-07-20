function g=sub_gradient(f,N,nV,nE,hyperedge_id)
% f=f+rand(size(f)).*1e-6;

v=f(hyperedge_id);
[v_max,v_max_ind]=max(v,[],"linear");
[v_min,v_min_ind]=min(v,[],"linear");
v_max_min=v_max-v_min;

index_max=hyperedge_id(v_max_ind);
index_min=hyperedge_id(v_min_ind);


g=zeros(nV,1);
for e=1:nE
    g(index_max(e))=g(index_max(e))+v_max_min(e);
    g(index_min(e))=g(index_min(e))-v_max_min(e);
end
g=g(N);

