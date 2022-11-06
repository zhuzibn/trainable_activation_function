function g = sigmoid(z,k,xc)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-k.*(z+xc)));
end
