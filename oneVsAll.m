function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
FirstTheta = zeros(n,1);
Opt = optimset('GradObj', 'on', 'MaxIter', 50);
     
for c = 1:num_labels
    temp=(y == c);
    temptheta(:,c) =  fmincg(@(t)(lrCostFunction(t,X,temp,lambda)),FirstTheta,Opt);
    all_theta(c,:)=temptheta(:,c)';
end












% =========================================================================


end
