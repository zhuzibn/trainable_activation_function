function g = Reluu(z,k,xc)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = max(0,k.*(z+xc));
%in z+xc,xc (25x1) is mapped to the same size with z (25x60000) before 
%the addition is performed, each column in z is identical
if (0)%test example
    
    %the result shows a 25x1 empty array
end

% the k.*(z+xc) performs the same operation
end