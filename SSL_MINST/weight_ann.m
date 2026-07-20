function [W]=weight_ann(data,num_s)
% the function to compute the weight matrix using kdtree

[m,n]=size(data);
kdtree = vl_kdtreebuild(data);
[idx, dist] = vl_kdtreequery(kdtree, data, data, 'NumNeighbors', num_s, 'MaxComparisons', min(n,2^10));
% The i-th column of idx/dist contains the nearest 512 points of the i-th point.
sigma=sparse([1:n],[1:n],1./max(dist(num_s,:),1e-2),n,n); 
id_row=repmat([1:n],num_s,1);
id_col=double(idx);
w=exp(-(dist*sigma).^2); 
W=sparse(id_row,id_col,w,n,n);