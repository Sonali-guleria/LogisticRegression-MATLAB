function [ pred ] = lr_pred( w, X )
%LR_PRED: Make prediction using weight vector w and feature matrix X
%   w is weight vector (d * 1)
%   x is feature matrix (d * n)
%   pred is binary prediction result based on w and X

pred = zeros(1,size(X,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Xnew = w'*X;
  sigX = sigmoid(Xnew);
  p = sigX;

 for i = 1:size(pred,2)
  
  if(p(i) > 0.5)
      pred(i) = 1;
  else
      pred(i) = 0;
  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

