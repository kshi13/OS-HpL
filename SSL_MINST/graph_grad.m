function [ G ] = graph_grad( W )
% Given eright matrix W, find the gradient matrix G, such that
% w_ij(u_i-u_j)=Gu, i.e., write the left hand side as a vector 

[m,~]=size(W);
n=m*(m-1)/2;
G=zeros(n,m);

row=0;
for i=1:m-1
%     d=diag(W,i)';
    d=W(i,i+1:end);
    row=row+1;
    G(row:row+(m-i-1),i)=d; % finished the positive part
    ind=sub2ind([n,m],row,i+1);
    G(ind:n+1:ind+n*(m-i)-1)=-d; % finished the negative part
    
    row=row+(m-i-1);
end
   
    
    
    



