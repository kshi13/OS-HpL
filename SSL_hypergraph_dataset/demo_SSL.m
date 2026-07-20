close all;
clear
addpath(genpath('data'));

num_L_all=[0.1,0.1,0.1,0.1,0.1;
    0.05,0.05,0.05,0.05,0.05];  % Labeling rate
num_run=10;
p=2;
Mean_error_OS=zeros(5,2); Std_error_OS=Mean_error_OS; Time_OS=Mean_error_OS;
Mean_error_GL=Mean_error_OS; Std_error_GL=Mean_error_OS; Time_GL=Mean_error_OS;
Mean_error_SGD=Mean_error_OS; Std_error_SGD=Mean_error_OS; Time_SGD=Mean_error_OS;
%% read data
for i_dataset=1:5
    i_dataset
    if i_dataset==1
        Y=readmatrix('node-labels-citeseer.txt');    HH = readmatrix('hyperedges-citeseer.txt');
    elseif i_dataset==2
        Y=readmatrix('node-labels-cora-author.txt'); HH = readmatrix('hyperedges-cora-author.txt');
    elseif i_dataset==3
        Y=readmatrix('node-labels-cora-cit.txt');    HH = readmatrix('hyperedges-cora-cit.txt');
    elseif i_dataset==4
        Y=readmatrix('node-labels-dblp.txt');        HH = readmatrix('hyperedges-dblp.txt');
    elseif i_dataset==5
        Y=readmatrix('node-labels-pubmed.txt');      HH = readmatrix('hyperedges-pubmed.txt');
    end
    HH(isnan(HH))=0; nV=length(Y); [nE,~]=size(HH);
    H=zeros(nV,nE);
    for i=1:nE
        edge_number=nnz(HH(i,:));
        edge=HH(i,1:edge_number);
        H(edge,i)=1;
    end
    I=sum(H,2)==0; H(I,:)=[]; Y(I)=[];  % remove isolated points
    [nV, nE]=size(H);
    HH=zeros(nE,size(HH,2));
    for i=1:nE
        temp=find(H(:,i));
        HH(i,1:length(temp))=temp;
    end
    W_full=sparse(H*H');  % the weight for graph Laplacian
    
    id_col=HH'; % each column is a hyperedge, for hypergraph Laplacian operator-splitting
    num_e=zeros(nV,1); % cardinality of each hyperedge
    for i=1:nE
        num_e(i)=nnz(HH(i,:));
    end
    hyperedge_id=id_col;  % each column is a hyperedge, for hypergraph Laplacian subgradient descent
    for i=1:nE
        hyperedge_id(1+num_e(i):end,i)=hyperedge_id(1,i);
    end
    
    %% start running three algorithms
    for i_label=1:2
        %% Graph Laplacian: Clique extension of the hypergaprh
        tic
        num_L=round(num_L_all(i_label,i_dataset)*nV);
        error=zeros(1,length(num_run));
        for run=1:num_run
            rand('seed',run);
            randnum=rand(nV,1);
            randnumsort=sort(randnum);
            mask=randnum<=randnumsort(num_L);
            id=find(mask);
            F=zeros(nV,max(Y));
            for i=1:max(Y) % number of classes
                Y0=Y;
                Y0(Y==i)=1; Y0(Y~=i)=-1; Y0=Y0.*mask;
                g=Y0(id);
                u=SSL_graph_Laplace(W_full,g,id);
                F(:,i)=u;
            end
            [max_F,index]=max(F,[],2);
            error(run)=sum(index~=Y)/nV;
        end
        Mean_error_GL(i_dataset,i_label)=mean(error)*100;
        Std_error_GL(i_dataset,i_label)=std(error*100);
        Time_GL(i_dataset,i_label)=toc;
        
        %% Hypergraph Laplacian Subgradient descent
        tic
        num_L=round(num_L_all(i_label,i_dataset)*nV);
        error=zeros(1,length(num_run));
        for run=1:num_run
            rand('seed',run);
            randnum=rand(nV,1);
            randnumsort=sort(randnum);
            mask=randnum<=randnumsort(num_L);
            id=find(mask);
            F=zeros(nV,max(Y));
            for i=1:max(Y) % number of classes
                Y0=Y;
                Y0(Y==i)=1; Y0(Y~=i)=-1; Y0=Y0.*mask;
                g=Y0(id);
                N=find(~Y0);
                u=F(:,i); u(id)=g;
                for iter=1:500    % Iter_SGD(i_label,i_dataset)
                    alpha=min(1,(0.16*iter)/1e5);
                    dt=1/((iter+1).^alpha);
                    u_N=u(N);
                    g1=sub_gradient(u,N,nV,nE,hyperedge_id);
                    u_N=u_N-dt*g1./(norm(g1)+0.00001);
                    u(N)=u_N;
                    u(id)=g;
                end
                F(:,i)=u;
            end
            [max_F,index]=max(F,[],2);
            error(run)=sum(index~=Y)/nV;
        end
        Mean_error_SGD(i_dataset,i_label)=mean(error)*100;
        Std_error_SGD(i_dataset,i_label)=std(error*100);
        Time_SGD(i_dataset,i_label)=toc;
        
        %% Hypergraph Laplacian Operator-splitting
        tic
        num_L=round(num_L_all(i_label,i_dataset)*nV);
        error=zeros(1,length(num_run));
        for run=1:num_run
            rand('seed',run);
            randnum=rand(nV,1);
            randnumsort=sort(randnum);
            mask=randnum<=randnumsort(num_L);
            id=find(mask);
            F=zeros(nV,max(Y));
            for i=1:max(Y) % number of classes
                Y0=Y;
                Y0(Y==i)=1; Y0(Y~=i)=-1; Y0=Y0.*mask;
                g=Y0(id);
                [u,energy]=SSL_HyperGp2L_OS_noweight(nV,id_col,g,id,p,num_e);
                F(:,i)=u;
            end
            [max_F,index]=max(F,[],2);
            error(run)=sum(index~=Y)/nV;
        end
        Mean_error_OS(i_dataset,i_label)=mean(error)*100;
        Std_error_OS(i_dataset,i_label)=std(error*100);
        Time_OS(i_dataset,i_label)=toc;
    end
end
Accuracy_GL=100-Mean_error_GL';
Accuracy_OS=100-Mean_error_OS';
Accuracy_SGD=100-Mean_error_SGD';

Std_GL=Std_error_GL;
Std_OS=Std_error_OS;
Std_SGD=Std_error_SGD;

Time_GL=Time_GL'/num_run;
Time_OS=Time_OS'/num_run;
Time_SGD=Time_SGD'/num_run;

% save('Accuracy_GL.mat','Accuracy_GL')
% save('Accuracy_OS.mat','Accuracy_OS')
% save('Accuracy_SGD.mat','Accuracy_SGD')
% 
% save('Std_GL.mat','Std_GL')
% save('Std_OS.mat','Std_OS')
% save('Std_SGD.mat','Std_SGD')
% 
% save('Time_GL.mat','Time_GL')
% save('Time_OS.mat','Time_OS')
% save('Time_SGD.mat','Time_SGD')



