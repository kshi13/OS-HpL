%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% projection on the l_1 norm ball
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% { x : l1norm(x)<=tau },
% where tau>=0 and
% l1norm = @(x) sum(abs(x(:))),
% of the N-D array y.

function x = projball_l1(y, tau)
	tmp = abs(y(:));
	if sum(tmp)<=tau, x = y; return; end
	lambda = max((cumsum(sort(tmp,1,'descend'))-tau)./(1:length(tmp))');
	x = y - max(min(y, lambda), -lambda);
end
