% run('C:\Program Files\MATLAB\vlfeat-0.9.21/toolbox/vl_setup')       %Laptop
% run('C:\Program Files\Polyspace\vlfeat-0.9.21/toolbox/vl_setup') %Desktop

addpath('./data');
addpath('./data/MNIST');
close all;clear;
rand('seed',1);
num_s=21;
% num_training=[35 350];
% num_trail=10;
num_training=[35];
num_trail=1;

Accuracy_GL=zeros(length(num_training),num_trail); Time_GL=zeros(length(num_training),num_trail);
Accuracy_HG_PD=Accuracy_GL; Time_HG_PD=Time_GL;
Accuracy_HG_OS=Accuracy_GL; Time_HG_OS=Time_GL;

[images, realLabels]=load_mnist; images=images/255;
realLabels(realLabels == 0) = 10;
[n, d] = size(images');

for jj=1:length(num_training)
    f = num_training(jj); % number of labeled data
    fprintf('Training number: %g, ', num_training(jj));
    for ii=1:num_trail
        fprintf('Trail number: %g, ', ii);
        id = randperm(n, f)';
        training_set= [id, realLabels(id)];
        %% graph Laplace
        fprintf('Graph Laplace');
        tic
        [W]=weight_ann(images,num_s);
        u_gl = zeros(n,10);
        for i = 1 : 10
            g=0.*id;
            g(training_set(:, 2) == i)=1;
            u_gl(:, i) = SSL_graph_Laplace(W, g, id);
        end
        [~, Labels] = max(u_gl');
        accuracy_GL=length(find(realLabels==Labels')) / n
        Accuracy_GL(jj,ii)=length(find(realLabels==Labels')) / n;
        Time_GL(jj,ii)=toc;
        
        %% hypergraph Laplace: Primal-Dual
        tic
        p=2;
        [W_full,id_col]=weight_hg_knn(images,num_s,p);
        u = zeros(n,10);
        for i = 1 : 10
            i
            g=zeros(size(training_set,1),1);
            subset = find(training_set(:, 2) == i);
            g(subset)=1;
            if sum(g)~=0
                training_points=training_set(:,1);
                u(:, i)=SSL_HyperGp2L_PD(W_full,id_col,g,training_points,num_s);
            end
        end
        [~, Labels] = max(u');
        accuracy_HG_PD=length(find(realLabels==Labels')) / n
        Accuracy_HG_PD(jj,ii)=length(find(realLabels==Labels')) / n;
        Time_HG_PD(jj,ii)=toc;
        
        %% hypergraph Laplace: operator-splitting
        tic
        p=2;
        [W_full,id_col]=weight_hg_knn(images,num_s,p);
        [Wij_all,W_Laplace_all,coe_matrix_fix_all]=initialization_OS(W_full,id_col,id);
        u_hg_os = zeros(n,10);
        for i = 1 : 10
            g=0.*id;
            g(training_set(:, 2) == i)=1;
            if max(g)==0
                u_hg_os(:, i)=0;
                energy=0;
            else
                [u_hg_os(:, i),energy]=SSL_HyperGp2L_OS(W_full,id_col,g,id,...
                                            Wij_all,W_Laplace_all,coe_matrix_fix_all,p);
            end
        end
        [~, Labels] = max(u_hg_os');
        accuracy_HG_OS=length(find(realLabels==Labels')) / n
        Accuracy_HG_OS(jj,ii)=length(find(realLabels==Labels')) / n;
        Time_HG_OS(jj,ii)=toc;
    end
end

% save('Accuracy_GL.mat','Accuracy_GL')
% save('Time_GL.mat','Time_GL')

% save('Accuracy_HG_OS.mat','Accuracy_HG_OS')
% save('Time_HG_OS.mat','Time_HG_OS')

% save('Accuracy_HG_PD.mat','Accuracy_HG_PD')
% save('Time_HG_PD.mat','Time_HG_PD')

