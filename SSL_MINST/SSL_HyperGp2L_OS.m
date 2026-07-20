function [u,energy]=SSL_HyperGp2L_OS(W_full,id_col,g,id,Wij_all,W_Laplace_all,coe_matrix_fix_all,p)

n=size(W_full,1);
u=zeros(n,1);
u(id)=g;

tau=1; 
mu=1/(tau);  
rho=1; 
sigma=(rho/p)^(1/(p-1));
k = (1:21^2)';
sigma_cof = sigma .*  (1 ./ (1 + sigma .* k));

hyperedge_degree=sum(id_col ~= 0, 1); % number of vertices in each hyperedge
mask_hyperedge=ismember(id_col,id); % label in hyperedge e
position_hyperedge=mask_hyperedge.*id_col;

id_e_c_all=cell(n,1); rhs_fix_all=id_e_c_all; d_all=id_e_c_all; %coe_matrix_fix_all=cell(n,1); 
for e=1:n
    d_all{e}=zeros(hyperedge_degree(e));
    id_e=mask_hyperedge(:,e); % label in hyperedge e
    id_e_c=~id_e;
    id_e_c_all{e}=id_e_c;
    I=position_hyperedge(:,e);
    I=I(I~=0);
    ge=u(I);
    W_Laplace=W_Laplace_all{e};
    rhs_fix_all{e}=W_Laplace*ge;
end
lambda_all=d_all;

idx_e = randperm(n);
for iter=1:100
    mu=mu*4;    
    u_old=u;
    for e=idx_e
        % e=randperm(n,1);
        d=d_all{e};
        lambda=lambda_all{e};
        id_e_c=id_e_c_all{e};
        Wij=Wij_all{e};  % w_{i,j}^{1/p} on each hyperedge non-symmetric
        I=id_col(:,e);
        ue=u(I);

        coe_matrix=coe_matrix_fix_all{e}+mu/rho*eye(nnz(id_e_c));
        rhs_fix=rhs_fix_all{e}+mu/rho*ue(id_e_c);
        %% u-subproblem
        w_v=(d-lambda/rho).*Wij;
        w_v=w_v.'-w_v;
        rhs=rhs_fix - sum(w_v(id_e_c,:),2);
        ue_c=coe_matrix\rhs;
        ue(id_e_c)=ue_c;
        
        %% d-subproblem
        Du=Wij.*(ue-ue.');
        Du_lambda=Du+lambda/rho;
        Du_lambda_proj=prox_L1_p2(Du_lambda,sigma_cof);
        d=Du_lambda-Du_lambda_proj;
        
        %% lambda
        lambda=lambda + rho*(Du-d);
        
        d_all{e}=d;
        lambda_all{e}=lambda;
        u(I)=ue;
    end
    % energy(iter)=Energy_HG(u,Wij_all,id_col,n,p);
    energy(iter)=0;
    error=sum(sum((u-u_old).^2))/sum(sum(u.^2))
    if error<5e-4
        break
    end   
end

