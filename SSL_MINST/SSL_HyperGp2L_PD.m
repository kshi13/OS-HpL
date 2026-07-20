function u=SSL_HyperGp2L_PD(W_full,id_col,g,id,num_s)
n=size(W_full,1);

% Hypergraph part
sigma=1/(3);
tau=1/(3*n);
m=num_s*(num_s-1)/2; % Length of K

k = (1:m)';
sigma_cof = sigma .*  (1 ./ (1 + sigma .* k));

u=zeros(n,1); % primal variable
alpha=zeros(num_s*(num_s-1)/2,n); % dual variable
K = graph_grad(ones(num_s,num_s));
ind_tril=tril(true(num_s,num_s),-1);  %ind_tril=find(ind_tril);
ind_p=find(K==1);  % index for 1 in K
[ind_nx,ind_ny]=find(K==-1);
[ind_nx_sort,sort_ind]=sort(ind_nx);
ind_ny_sort=ind_ny(sort_ind);
ind_n=sub2ind(size(K),ind_nx_sort,ind_ny_sort);  % index for -1 in K

Kp=K; Kp(K==-1)=0; ind_up=Kp*([1:num_s]');
Kn=K; Kn(K==1)=0; ind_un=-Kn*([1:num_s]');  % index for Keue


K_alpha_bar=zeros(n,1);
Ke=K;
I_old=id_col(:,1); K_alpha_diff_old=zeros(num_s,1);

for iter=1:100
    u_old=u;
    idx_e = randperm(n);
    for e=idx_e
        % update the primal variable
        u=u-tau*(K_alpha_bar);
        u(id)=g;
        % update the dual variable
        % e=randperm(n,1);
        I=id_col(:,e);
        ue=u(I);
        Wij=W_full(I,I);  % non-symmetric
        Wij=double(Wij)/2^8;
        Wijt=Wij'; coef_K=Wijt(ind_tril);
        Ke(ind_p)=coef_K; Ke(ind_n)=-coef_K; % replace 1,-1 in K with the weight

        Keue=coef_K.*(ue(ind_up)-ue(ind_un)); % ==Keue
        beta=alpha(:,e)+sigma*Keue;
        alpha_e=prox_L1_p2(beta,sigma_cof); 

        alpha_diff=alpha_e-alpha(:,e);
        alpha(:,e)=alpha_e;

        % Extrapolation
        K_alpha_diff=Ke'*alpha_diff;
        K_alpha_bar(I)=K_alpha_bar(I)+(1+n)*K_alpha_diff;
        K_alpha_bar(I_old)=K_alpha_bar(I_old)-n*K_alpha_diff_old;
        I_old=I; K_alpha_diff_old=K_alpha_diff;
    end
    error=sum(sum((u-u_old).^2))/sum(sum(u.^2));
    if error<5e-4
        break
    end
end % end of primal-dual algorithm


