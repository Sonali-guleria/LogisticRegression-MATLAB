function [ f, g ] = fv_grad( w_curr, X, y )
%FV_GRAD: returns the objective function value f and the gradient g w.r.t w
%at point w_curr
%   w_curr is current weight vector (d * 1)
%   X is feature values (d * n)
%   y is label (1 * n)

% Set parameters
f = 0;
g = zeros(length(w_curr), 1);
lambda_val = 25;
lambda = ones(length(w_curr), 1) * lambda_val;
lambda(1) = 0;  % do not penalize weight associated with the intercept term


xNew = w_curr' * X;
sigX = sigmoid(xNew);

f = -(sum(y * log(sigX)' + (1-y) * (log(1-sigX))'));
%+sum((lambda./2)*sum(dot(w_curr,w_curr)));
g = (y-sigX)*X';
g = g';
%g = g + lambda .* w_curr;

end

