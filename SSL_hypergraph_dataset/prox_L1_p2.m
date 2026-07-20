function solution = prox_L1_p2_OS(beta, sigma)

b = abs(beta);
bs = sort(b(:), 'descend');
m=length(bs);
cs = cumsum(bs);

k = (1:m)';
lambda_k = sigma .* cs .* (1 ./ (1 + sigma .* k));

idx = find(lambda_k >= bs, 1, 'first');  
if isempty(idx)
    lambda = 0;
else
    lambda = lambda_k(idx);
end
solution = max(b - lambda, 0);
solution = solution .* sign(beta);


