function [energy]=Energy_HG(u,Wij_all,id_col,n,p)


energy=0;
% W_full_sort=W_full_sort.^(1/p);
for e=1:n %randperm(n,500) %randperm(n)%
    I=id_col(:,e);
    ue=u(I);
    Wij=Wij_all{e};  
    Due=Wij.*(ue-ue');

    energy=energy+(max(max(abs(Due))))^p;
end





