function [J, Grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
Grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
theta(1) = 0;

Hypo= sigmoid(X.*theta);
J =-mean(y.*log(Hypo)+(1-y).*log(1-Hypo))+(lambda/2).*mean(theta(2:end).^2);

Grad(1,1)=(1/m)*(X(:,1)'*(Hypo-y)); 
Grad(2:end,1)=((1/m)*(X(:,2:end))'*(Hypo-y))+(lambda/m)*theta(2:end);
% =============================================================

Grad = Grad(:);

end
