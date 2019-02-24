function p = predict(Theta1,Theta2,X)
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);
A1 = [ones(m,1) X];
A2 = sigmoid(A1*Theta1');
A2 = [ones(m,1) A2];
A3 = sigmoid(A2*Theta2');
[~,p]= max(A3,[],2);
end
