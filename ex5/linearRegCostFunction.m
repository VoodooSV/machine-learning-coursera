function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

    h = X * theta; % [12,1] = [12,2] * [2,1]
    error = h - y; % [12,1]
    error_sqr = error .^ 2; % [12, 1]
    theta(1) = 0;
    J = 1/(2*m) * sum(error_sqr); % [1,1], Unregularized cost
    J = J + lambda/(2*m) * sum(theta .^ 2); % [1,1]

    grad = X' * error; % [2,1] = [2,12] * [12,1], Unregularized gradient
    grad = 1/m * grad + lambda/m * theta; % [2,1]





% =========================================================================

grad = grad(:);

end
