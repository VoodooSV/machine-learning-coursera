function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part1

% ex4 Tutorial for forward propagation and cost
% https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog
   
%     y_matrix = full(ind2vec(y', num_labels)');
    % or:
%     y_matrix = [1:num_labels] == y;
    % or:
    eye_matrix = eye(num_labels);
    y_matrix = eye_matrix(y, :);

    a1 = [ones(m, 1) X]; % [m, n+1], Add ones to the X data matrix
    z2 = a1 * Theta1'; % [m, 25] = [m, n+1] * [n+1, 25]
    
    sigmoid_z2 = sigmoid(z2); % [m, 25]
    a2 = [ones(m, 1) sigmoid_z2]; % [m, 26], Add ones to the data matrix
    z3 = a2 * Theta2'; % [m, K] = [m, 26] * [26, K]
    sigmoid_z3 = sigmoid(z3); % [m, K]
    hypothesis = sigmoid_z3; % [m, K]
    
    temp_sum = 0;
    for i=1:num_labels
        temp_sum = temp_sum + sum(-y_matrix(1:end, i) .* log(hypothesis(1:end, i)) - (1 - y_matrix(1:end, i)) .* log(1 - hypothesis(1:end, i)));
    end;

    J = 1/m * temp_sum;

    % Computing the NN cost J using the matrix product
    % https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/AzIrrO7wEeaV3gonaJwAFA

%     hypothesis = hypothesis >= 0.5; % Convert to logical values [0, 1]
%     hypothesis = double(hypothesis); % Convert to double from logical values

    % Regularization
    % Must set first columns to ZERO, otherwise I got wrong result ( 0.383770)
%    Theta1(:,1) = 0;
%    Theta2(:,1) = 0;
%    regularization = lambda/(2*m) * (sum(sum(power(Theta1, 2))) + sum(sum(power(Theta2, 2))));
    % or exclude first columns from computation
    regularization = lambda/(2*m) * (sum(sum(power(Theta1(:, 2:end), 2))) + sum(sum(power(Theta2(:, 2:end), 2))));
    
    J = J + regularization;

% Part 2. Backpropagation

% ex4 tutorial for nnCostFunction and backpropagation
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q

    d3 = hypothesis - y_matrix; % [m, K] 
    d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); % [m, 25] = [m, K] * [K, 25] .* [m, 25]
    Delta1 = d2' * a1; % [25, n+1] = [25, m] * [m, n+1]
    Delta2 = d3' * a2; % [K, 26] = [K, m] * [m, 26]
    
    Theta1_grad = 1/m * Delta1;
    Theta2_grad = 1/m * Delta2;

% Part 3. Cost Regularization

    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
