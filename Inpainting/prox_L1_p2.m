function solution = prox_L1_p2(beta, sigma_cof)

b = abs(beta);
bs = sort(b(:), 'descend');
cs = cumsum(bs);

lambda_k=cs .*sigma_cof;
idx = find(lambda_k >= bs, 1, 'first');  
if isempty(idx)
    lambda = 0;
else
    lambda = lambda_k(idx);
end

solution = max(b - lambda, 0);
solution = solution .* sign(beta);


