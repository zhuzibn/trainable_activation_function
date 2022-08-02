%% Initialization
clear; close all; clc;tic
ddebug=0;
%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
% (note that we have mapped "0" to label 10)
modee=3;%1:x 2:k(x+c)  3:kx  4:kx+c
firstlayeraf=1;%the activation function of the first layer is 1:sigmoid 2:relu
find_max_weight=0;
discretization=0;%1/0: discretize/not-discretize the weight
nonlinearity=0;%1/0: weight update is nonlinear/linear
oss=1;%1:windows 0 linux
rng('shuffle');

if oss
addpath('D:\onedrive\projects\2021-trainable spintronic neuron-xin yue\manuscript\code\data')
else
addpath('/public/home/zhuzf/code/project/xinyue/data')    
end

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
    %save('tmp.mat','initial_Theta1','initial_Theta2','initial_k1','initial_k2','initial_c1','initial_c2')
    load('savedweights.mat') ;
    %load('savedweights.mat');
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
num_iters=1000;
alpha = 0.3;
lambdaa=0;%regularization constant
J_ = zeros(num_iters, 1);
a0=[ones(m,1) X];
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
    
    switch modee
        case 1
            k1=ones(size(k1));
            k2=ones(size(k2));
            xc1=zeros(size(xc1));
            xc2=zeros(size(xc2));
        case 2
        case 3
            xc1=zeros(size(xc1));
            xc2=zeros(size(xc2));
    end
    %% forward propagation
    z1=w1*a0';
    switch firstlayeraf
        case 1
            a1=sigmoid(z1,k1,xc1);
        case 2
            a1=Reluu(z1,k1,xc1);
    end
    a1=[ones(1,m);a1];
    z2=w2*a1;
    a2=sigmoid(z2,k2,xc2);
    
    switch modee
        case 1
            J(iter)=sum(sum(-y_tmp.*log(a2)-(1-y_tmp).*log(1-a2)))/m+...
                lambdaa/(2*m)*(sum(sum(w1.^2))+sum(sum(w2.^2)));
        case 2
            J(iter)=sum(sum(-y_tmp.*log(a2)-(1-y_tmp).*log(1-a2)))/m+...
                lambdaa/(2*m)*(sum(sum(w1.^2))+sum(sum(w2.^2))+sum(sum(k1.^2))+sum(sum(k2.^2))+sum(sum(xc1.^2))+sum(sum(xc2.^2)));
        case 3
            J(iter)=sum(sum(-y_tmp.*log(a2)-(1-y_tmp).*log(1-a2)))/m+...
                lambdaa/(2*m)*(sum(sum(w1.^2))+sum(sum(w2.^2))+sum(sum(k1.^2))+sum(sum(k2.^2)));
    end

    [iter,J(iter)];
    
    %% backward propagation   
    switch firstlayeraf
        case 1
        A1a=z1+xc1;
        A1b=exp(-k1.*A1a);
        da1dz1=(k1.*A1b)./((1+A1b).^2);
        da1dk1=(A1a.*A1b)./((1+A1b).^2);
        da1dxc1=(k1.*A1b)./((1+A1b).^2);
        clear A1a A1b
        case 2
            tmp=k1.*(z1+xc1);
            da1dz1=zeros(size(tmp));
            da1dk1=zeros(size(tmp));
            da1dxc1=zeros(size(tmp));
            tmp_pos=tmp>0;
            tmp1=repmat(k1,1,size(da1dz1,2));
            da1dz1(tmp_pos)=tmp1(tmp_pos);
            tmp2=z1+xc1;
            da1dk1(tmp_pos)=tmp2(tmp_pos);
            da1dxc1(tmp_pos)=tmp1(tmp_pos);
            clear tmp tmp1 tmp2
    end
        
        A2a=z2+xc2;
        A2b=exp(-k2.*A2a);
                
        da2dz2=(k2.*A2b)./((1+A2b).^2);
        da2dk2=(A2a.*A2b)./((1+A2b).^2);
        da2dxc2=(k2.*A2b)./((1+A2b).^2);
               
        clear A2a A2b
        
        dAL=-(y_tmp./a2-(1-y_tmp)./(1-a2));

        dcdw2=(dAL.*da2dz2)*a1';
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end,:);
        dcdw1=(tmp1.*da1dz1)*a0;
        clear tmp tmp1
        
        dcdk2=dAL.*da2dk2;
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end,:);
        dcdk1=tmp1.*da1dk1;
        clear tmp tmp1
        
        dcdxc2=dAL.*da2dxc2;
        
        tmp=w2'*(dAL.*da2dz2);
        tmp1=tmp(2:end,:);
        dcdxc1=tmp1.*da1dxc1;
        
    dcdw2ave=dcdw2/m;        
    dcdw2ave(:,2:end)=dcdw2ave(:,2:end)+2*lambdaa*w2(:,2:end)/m;%change to zeros(?)
    
    dcdw1ave=dcdw1/m;
    dcdw1ave(:,2:end)=dcdw1ave(:,2:end)+2*lambdaa*w1(:,2:end)/m;
    
    dcdk2ave=sum(dcdk2,2)/m;
    dcdk2ave(:,2:end)=dcdk2ave(:,2:end)+2*lambdaa*k2(:,2:end)/m;
    %the regularization is also problematic, but since it only has one column, the result is correct
    
    dcdk1ave=sum(dcdk1,2)/m;
    dcdk1ave(:,2:end)=dcdk1ave(:,2:end)+2*lambdaa*k1(:,2:end)/m;
    
    dcdxc2ave=sum(dcdxc2,2)/m;
    dcdxc2ave(:,2:end)=dcdxc2ave(:,2:end)+2*lambdaa*xc2(:,2:end)/m;
    
    dcdxc1ave=sum(dcdxc1,2)/m;
    dcdxc1ave(:,2:end)=dcdxc1ave(:,2:end)+2*lambdaa*xc1(:,2:end)/m;

    if nonlinearity
        G1max=Theta1_max_pos-Theta1_max_neg_;
        G1min=0;        
        dcdw1ave=nonlinearG(G1max,G1min,nonlinear_fac,w1,dcdw1ave);
        
        G2max=Theta2_max_pos-Theta2_max_neg_;
        G2min=0;
        dcdw2ave=nonlinearG(G2max,G2min,nonlinear_fac,w2,dcdw2ave);
        if (0)%plot the nonlinear curve
            P_=linspace(0,1,100);
            
            [G_i_Theta1,G_d_Theta1]=nonlinearG_plot(G1max,G1min,nonlinear_fac,P_);
            [G_i_Theta1_linear,G_d_Theta1_linear]=nonlinearG_plot(G1max,G1min,0,P_);
            
            [G_i_Theta2,G_d_Theta2]=nonlinearG_plot(G2max,G2min,nonlinear_fac,P_);
            [G_i_Theta2_linear,G_d_Theta2_linear]=nonlinearG_plot(G2max,G2min,0,P_);
            
            figure;hold on
            plot(P_,G_i_Theta1_linear)
            plot(P_,G_d_Theta1_linear)
            xlabel('Pulse');ylabel('G')
            
        end
    end
    
    w2=w2-alpha*dcdw2ave;
    w1=w1-alpha*dcdw1ave;
    k2=k2-alpha*dcdk2ave;
    k1=k1-alpha*dcdk1ave;
    xc2=xc2-alpha*dcdxc2ave;
    xc1=xc1-alpha*dcdxc1ave;   
    
    if discretization
        w1 = interp1(roundTheta1,roundTheta1,w1,'nearest');
        w2 = interp1(roundTheta2,roundTheta2,w2,'nearest');
    end
    %find max weight
    if find_max_weight
        Theta1_pos_max_tmp=max(w1(w1>0));
        if ~isempty(Theta1_pos_max_tmp)
            Theta1_pos_max=max(Theta1_pos_max_tmp,Theta1_pos_max);
        end
        
        Theta1_neg_max_tmp=max(-w1(w1<0));
        if ~isempty(Theta1_neg_max_tmp)
            Theta1_neg_max=max(Theta1_neg_max_tmp,Theta1_neg_max);
        end
        
        Theta2_pos_max_tmp=max(w2(w2>0));
        if ~isempty(Theta2_pos_max_tmp)
            Theta2_pos_max=max(Theta2_pos_max_tmp,Theta2_pos_max);
        end
        
        Theta2_neg_max_tmp=max(-w2(w2<0));
        if ~isempty(Theta2_neg_max_tmp)
            Theta2_neg_max=max(Theta2_neg_max_tmp,Theta2_neg_max);
        end
        
    end
end
switch modee
    case 1
        k1=ones(size(k1));
        k2=ones(size(k2));
        xc1=zeros(size(xc1));
        xc2=zeros(size(xc2));
    case 2
    case 3
        %xc1=zeros(size(xc1));
        %xc2=zeros(size(xc2));
end
Theta1_neg_max=-Theta1_neg_max;
Theta2_neg_max=-Theta2_neg_max;

%% Implement Predict
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(w1, w2, k1,k2,xc1,xc2,Xtest,firstlayeraf);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
predic=mean(double(pred == ytest)) * 100;

toc
%save('final.mat','predic');
if (0)
    plot(iterplot,J,'*')
    xlabel('iterations');ylabel('J')
end

