%% Initialization
clear; close all; clc;tic
ddebug=0;
%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
% (note that we have mapped "0" to label 10)
find_max_weight=0;
discretization=0;%1/0: discretize/not-discretize the weight
nonlinearity=0;%1/0: weight update is nonlinear/linear

Theta1_max_pos=1.42; %these max values are obtained by running 1000 iterations
Theta1_max_neg_=-1.69;
Theta2_max_pos=0;
Theta2_max_neg_=-5.4;

if discretization
    discrete_bits=8;
    discrete_level=2^discrete_bits;
    roundTheta1=linspace(Theta1_max_neg_,Theta1_max_pos,discrete_level);
    roundTheta2=linspace(Theta2_max_neg_,Theta2_max_pos,discrete_level);
end

if nonlinearity
    nonlinear_fac=1;
    if nonlinear_fac==0
        error('in the nonlinear mode, do not set nonlinear_fac to zero')
        %In the nonlinear mode, you should set nonlinearity=0
    end
end

load('Train.mat');
y=cell2mat(Train(:,2))+1;
m = size(y, 1);%No. of samples
Xtmp=cell2mat(Train(:,1))';
X=reshape(Xtmp,input_layer_size,m)'/1000;%how does the array listed in ex4data1.mat?
clear Xtmp

load('Test.mat');
ytest=cell2mat(Test(:,2))+1;
mtest = size(ytest, 1);%No. of samples
Xtmp=cell2mat(Test(:,1))';
Xtest=reshape(Xtmp,input_layer_size,mtest)'/1000;%how does the array listed in ex4data1.mat?
clear Xtmp

if (0)
    % Randomly select 100 data points to display
    sel = randperm(size(X, 1));
    sel = sel(1:100);
    displayData(X(sel, :));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

if ddebug
    load('savedweights.mat') ;
    w1=initial_Theta1;
    k1=initial_k1;
    xc1=initial_c1;
    w2=initial_Theta2;
    k2=initial_k2;
    xc2=initial_c2;
else
    w1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    k1 = randInitializeWeights(0, hidden_layer_size);
    xc1 = randInitializeWeights(0, hidden_layer_size);
    w2 = randInitializeWeights(hidden_layer_size, num_labels);
    k2 = randInitializeWeights(0, num_labels);
    xc2 = randInitializeWeights(0, num_labels);
    %save('storedrandomweights.mat','Theta1','Theta2')
    %load('storedrandomweights.mat');
end

y_tmp=zeros(num_labels,m);
for ct2=1:m
    if y(ct2)==1
        y_tmp(:,ct2)=[1;zeros(num_labels-1,1)];
    elseif y(ct2)==num_labels
        y_tmp(:,ct2)=[zeros(num_labels-1,1);1];
    else
        y_tmp(:,ct2)=[zeros(y(ct2)-1,1);1;zeros(num_labels-y(ct2),1)];
    end
end
%% Training NN
%  You have now implemented all the code necessary to train a neural
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

%% gradient descent
num_iters=10;
alpha = 0.3;
J_ = zeros(num_iters, 1);
a0_=[ones(m,1) X];
iterplot=[1:num_iters];

Theta1_pos_max=0;
Theta1_neg_max=0;
Theta2_pos_max=0;
Theta2_neg_max=0;

for iter = 1:num_iters
    dcdw2ave=zeros(size(w2));%average
    dcdw1ave=zeros(size(w1));
    dcdk2ave=zeros(size(k2));
    dcdk1ave=zeros(size(k1));
    dcdxc2ave=zeros(size(xc2));
    dcdxc1ave=zeros(size(xc1));
    
    %% forward propagation
    z1=w1*a0_';
    a1=sigmoid(z1,k1,xc1);
    a1=[ones(1,m);a1];
    z2=w2*a1;
    a2=sigmoid(z2,k2,xc2);
    
    J(iter)=sum(sum(-y_tmp.*log(a2)-(1-y_tmp).*log(1-a2)))/m;
    
    [iter,J(iter)]
    
    %% backward propagation
    dcdw2tot=0;
    dcdw1tot=0;
    dcdk2tot=0;
    dcdk1tot=0;
    dcdxc2tot=0;
    dcdxc1tot=0;
    for ct1=1:m
        a0=a0_(ct1,:);
        z1=w1*a0';
        a1=sigmoid(z1,k1,xc1);
        a1=[1;a1];
        z2=w2*a1;
        a2=sigmoid(z2,k2,xc2);
        
        A1a=z1+xc1;
        A1b=exp(-k1.*(z1+xc1));
        
        da1dz1=k1.*A1b./(1+A1b).^2;
        da1dk1=A1a./(1+A1b).^2.*A1b;
        da1dxc1=k1.*A1b./(1+A1b).^2;
        
        A2a=z2+xc2;
        A2b=exp(-k2.*(z2+xc2));
        
        da2dz2=k2.*A2b./(1+A2b).^2;
        da2dk2=A2a./(1+A2b).^2.*A2b;
        da2dxc2=k2.*A2b./(1+A2b).^2;
        clear A1a A1b A2a A2b
        
        dAL=-(y_tmp(:,ct1)./a2-(1-y_tmp(:,ct1))./(1-a2));
        
        dcdw2=dAL.*da2dz2*a1';
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end);
        dcdw1=tmp1.*da1dz1*a0;
        clear tmp tmp1
        
        dcdk2=dAL.*da2dk2;
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end);
        dcdk1=tmp1.*da1dk1;%check if this is correct
        clear tmp tmp1
        
        dcdxc2=dAL.*da2dxc2;
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end);
        dcdxc1=tmp1.*da1dxc1;
        clear tmp tmp1
        
        dcdw2tot=dcdw2tot+dcdw2;
        dcdw1tot=dcdw1tot+dcdw1;
        dcdk2tot=dcdk2tot+dcdk2;
        dcdk1tot=dcdk1tot+dcdk1;
        dcdxc2tot=dcdxc2tot+dcdxc2;
        dcdxc1tot=dcdxc1tot+dcdxc1;
    end
    dcdw2ave=dcdw2tot/m;
    dcdw2ave(:,2:end)=dcdw2ave(:,2:end);%change to zeros(?)
    
    dcdw1ave=dcdw1tot/m;
    dcdw1ave(:,2:end)=dcdw1ave(:,2:end);
    
    dcdk2ave=dcdk2tot/m;
    dcdk2ave(:,2:end)=dcdk2ave(:,2:end);
    
    dcdk1ave=dcdk1tot/m;
    dcdk1ave(:,2:end)=dcdk1ave(:,2:end);
    
    dcdxc2ave=dcdxc2tot/m;
    dcdxc2ave(:,2:end)=dcdxc2ave(:,2:end);
    
    dcdxc1ave=dcdxc1tot/m;
    dcdxc1ave(:,2:end)=dcdxc1ave(:,2:end);
    
    w2=w2-alpha*dcdw2ave;
    w1=w1-alpha*dcdw1ave;
    k2=k2-alpha*dcdk2ave;
    k1=k1-alpha*dcdk1ave;
    xc2=xc2-alpha*dcdxc2ave;
    xc1=xc1-alpha*dcdxc1ave;
end
Theta1_neg_max=-Theta1_neg_max;
Theta2_neg_max=-Theta2_neg_max;

if (0)
    %% Visualize Weights
    %  You can now "visualize" what the neural network is learning by
    %  displaying the hidden units to see what features they are capturing in
    %  the data.
    
    fprintf('\nVisualizing Neural Network... \n')
    
    displayData(w1(:, 2:end));
    
    fprintf('\nProgram paused. Press enter to continue.\n');
end
%% Implement Predict
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(w1, w2, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
predic=mean(double(pred == ytest)) * 100;
%save('final.mat');
toc

if (0)
    plot(iterplot,J,'*')
    xlabel('iterations');ylabel('J')
end

