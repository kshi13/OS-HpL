function [W_full,id_col]=weight_hg_knn(data,num_s,p)

[m,n]=size(data);

kdtree = vl_kdtreebuild(data);
[idx, dist] = vl_kdtreequery(kdtree, data, data, 'NumNeighbors', 1024, 'MaxComparisons', min(n,2^10));

% The i-th column of idx/dist contains the nearest 20 points of the i-th point.
sigma=sparse([1:n],[1:n],1./max(dist(num_s,:),1e-2),n,n); %

id_col_full=double(idx);
w=exp(-(dist*sigma).^2); 
w=w.^(1/p);
id_col=id_col_full(1:num_s,:);

W_full=uint8(w*2^8);
W_full_sort=zeros(n,n,'uint8');
for i=1:n
    W_full_sort(i,id_col_full(:,i))=W_full(:,i); % W_ij: simility between i and j
end
W_full=W_full_sort;


    