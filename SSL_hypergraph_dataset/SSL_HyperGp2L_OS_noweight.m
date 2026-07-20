function [u,energy]=SSL_HyperGp2L_OS_noweight(n,id_col,g,id,p,num_e)
energy=0;
nE=size(id_col,2);
u=zeros(n,1);
u(id)=g;

tau=1;  
mu=1/(1*tau); 
rho=1; 
sigma=(rho/p)^(1/(p-1));

hyperedge_degree=sum(id_col ~= 0, 1); % number of vertices in each hyperedge
mask_hyperedge=ismember(id_col,id); % label in hyperedge e
position_hyperedge=mask_hyperedge.*id_col;

id_e_c_all=cell(nE,1);  d_all=id_e_c_all;
sum_ge_all=zeros(nE,1); num_id_e_all=sum_ge_all;
for e=1:nE
    d_all{e}=zeros(hyperedge_degree(e));
    id_e=mask_hyperedge(1:num_e(e),e); % label in hyperedge e
    id_e_c=~id_e;
    id_e_c_all{e}=id_e_c;
    I=position_hyperedge(1:num_e(e),e);
    I=I(I~=0);
    ge=u(I);
    sum_ge_all(e)=2*sum(ge);
    num_id_e_all(e)=nnz(id_e);
end
lambda_all=d_all;

idx_e = randperm(nE);
for iter=1:100
    mu=mu*1.2;
    murho=mu/rho;
    u_old=u;
    for e=idx_e
        d=d_all{e};
        lambda=lambda_all{e};
        id_e_c=id_e_c_all{e};
        sum_ge=sum_ge_all(e);
        num_id_e=num_id_e_all(e);
        I=id_col(1:num_e(e),e);
        ue=u(I);
        %% u-subproblem
        v=d-lambda/rho;
        v=v-v.';
        if num_id_e==0
            h=sum_ge+sum(v,2)+murho*ue;
            ue=1/(2* num_e(e) +murho) * (  h + 1/(murho/2+num_id_e) * sum(h)  );
        else
            h=sum_ge+sum(v(id_e_c,:),2)+murho*ue(id_e_c);
            ue_c=1/(2* num_e(e) +murho) * (  h + 1/(murho/2+num_id_e) * sum(h)  );
            ue(id_e_c)=ue_c;
        end
        
        %% d-subproblem
        Du=ue-ue.';
        Du_lambda=Du+lambda/rho;
        Du_lambda_proj=prox_L1_p2(Du_lambda,sigma);
        d=Du_lambda-Du_lambda_proj;
        
        %% lambda
        lambda=lambda + rho*(Du-d);
        d_all{e}=d;
        lambda_all{e}=lambda;
        u(I)=ue;
    end
    %     energy(iter)=Energy_HG(u,id_col,nE,p,num_e);
    error=sum(sum((u-u_old).^2))/sum(sum(u.^2));
    if error<5e-5
        break
    end
end

