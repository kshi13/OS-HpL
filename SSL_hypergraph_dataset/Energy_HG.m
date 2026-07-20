function [energy]=Energy_HG(u,id_col,nE,p,num_e)


energy=0;
% W_full_sort=W_full_sort.^(1/p);
for e=1:nE %randperm(n,500) %randperm(n)%
    I=id_col(1:num_e(e),e);
    ue=u(I);
    Due=(ue-ue');

    energy=energy+(max(max(abs(Due))))^p;
end





