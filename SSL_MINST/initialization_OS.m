function [Wij_all,W_Laplace_all,coe_matrix_fix_all]=initialization_OS(W_full,id_col,id)
% output: Wij_all: w_{i,j}^{1/p} on each hyperedge
%         W_Laplace_all: w_{i,j}^{2/p}+ w_{j,i}^{2/p} on each hyperedge


n=size(W_full,1);
Wij_all=cell(n,1); 
W_Laplace_all=Wij_all;

coe_matrix_fix_all=Wij_all;
mask_hyperedge=ismember(id_col,id); % label in hyperedge e

for e=1:n
    id_e=mask_hyperedge(:,e); % label in hyperedge e
    id_e_c=~id_e;

    I=id_col(:,e);
    temp=double(W_full(I,I))/(2^8);
    Wij_all{e}=temp;
    temp_2=temp.*temp;
    W_Laplace=temp_2+temp_2.';
    W_Laplace_all{e}=W_Laplace(id_e_c,id_e);  

    coe_matrix_fix_all{e}=diag(sum(W_Laplace(id_e_c,:),2))-W_Laplace(id_e_c,id_e_c); 
end



