function u=SSL_graph_Laplace(W,g,id)
% W:       weight matrix;
% g:       labeled value;
% id:      index of labeled points;

n=size(W,1);
u=zeros(n,1);
u(id)=g;
id_c=setdiff(1:n, id);
u_c=u(id_c);


W_Laplace=W+W';

coe_matrix=diag(sum(W_Laplace(id_c,:),2))-W_Laplace(id_c,id_c);

rhs=W_Laplace(id_c,id)*g;

L = ichol(coe_matrix);
u_c=pcg(coe_matrix,rhs,1e-4,100,L,L',u_c);
u(id_c)=u_c;
