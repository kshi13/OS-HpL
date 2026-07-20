function [W_full,id_col]=weight_hg_knn(f,px_h,py_h,local_scale,num_s,p)

[n1,n2]=size(f); n=n1*n2;

x1=[1:1:n1]; x2=[1:1:n2];
[X,Y]=meshgrid(x1,x2);
X1=reshape(X,[],1); X2=reshape(Y,[],1);

X1_c=X1*255/n1;
X2_c=X2*255/n2;

up1=image2patch_center(f,X1,X2,px_h,py_h);
up=[up1,local_scale*X1_c,local_scale*X2_c];

%%
data=up';  % each column is a point
kdtree = vl_kdtreebuild(data);
[idx, dist] = vl_kdtreequery(kdtree, data, data, 'NumNeighbors', 1024, 'MaxComparisons', min(n,2^10));

% The i-th column of idx/dist contains the nearest 50 points of the i-th
% point.
sigma=sparse([1:n],[1:n],1./max(dist(num_s,:),1e-2),n,n); %

id_row_full=repmat([1:n],1024,1);
id_col_full=double(idx);
w=exp(-(dist*sigma).^2); 
w=w.^(1/p);
% W=sparse(id_row_full,id_col_full,w,n,n);
id_col=id_col_full(1:num_s,:);


W_full=uint8(w*2^8);
W_full_sort=zeros(n,n,'uint8');
for i=1:n
    W_full_sort(i,id_col_full(:,i))=W_full(:,i); % W_ij: simility between i and j
end
W_full=W_full_sort;