load lr_train.mat;

% initialize
epsilon = 1e-6;
num_iter = 2;

w = rand(size(train.X, 1), 1) * 1e-4;
d = length(w);
err = zeros(2, 1);

[f, g] = fv_grad(w, train.X, train.y);

for i = 1 : num_iter
    grad_approx = zeros(d, 1);
    for j = 1 : d

        thetaPlus = w;
        thetaPlus(j) = thetaPlus(j) + epsilon;
        thetaMinus = w;
        thetaMinus(j) = thetaMinus(j) - epsilon;
        [plusf,plusg] = fv_grad(thetaPlus, train.X, train.y);
        [minusf,minusg] = fv_grad(thetaMinus, train.X, train.y);
        jthetaPlus = plusf;
        jthetaMinus = minusf;
        grad_approx(j)= (plusf - minusf) ./ (2*epsilon);
   
    end
    
    err(i) = 1 / d * sum(abs(abs(grad_approx) - abs(g)));

end


if mean(err) < 1e-6
    fprintf('pass!')
else
    fprintf('fail.')
end
