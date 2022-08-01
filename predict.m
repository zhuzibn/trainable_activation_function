function p = predict(Theta1, Theta2,k1,k2,xc1,xc2, X,firstlayeraf)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);
    switch firstlayeraf
        case 1
            h1 = sigmoid(Theta1*[ones(m, 1) X]',k1,xc1);
        case 2
            h1 = Reluu(Theta1*[ones(m, 1) X]',k1,xc1);
    end
h1=[ones(1,m);h1];
h2 = sigmoid(Theta2*h1,k2,xc2);
[dummy, p] = max(h2', [], 2);

% =========================================================================


end
