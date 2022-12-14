%% Initialization
clear; close all; clc;tic
ddebug=0;
%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
% (note that we have mapped "0" to label 10)
modee=2;%1:x 2:k(x+c)  3:kx  4:kx+c (to do)
firstlayeraf=1;%the activation function of the first layer is 1:sigmoid 2:relu
find_max_weight=1;
discretization=0;%1/0: discretize/not-discretize the weight
nonlinearity=0;%1/0: weight update is nonlinear/linear
oss=0;%1:windows 0 linux
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
iter_output=100;%save the prediction accuracy every #iter_output
alpha = 0.3;
a0=[ones(m,1) X];
iterplot=linspace(1,num_iters,floor(num_iters/iter_output));

predic_=zeros(1,floor(num_iters/iter_output));
J_=zeros(1,floor(num_iters/iter_output));
if find_max_weight %give initial value
    w1_pos_max=0;
    w1_neg_max=0;
    w2_pos_max=0;
    w2_neg_max=0;
    k1_pos_max=0;
    k1_neg_max=0;
    k2_pos_max=0;
    k2_neg_max=0;
    xc1_pos_max=0;
    xc1_neg_max=0;
    xc2_pos_max=0;
    xc2_neg_max=0;
end
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
for iter = 1:num_iters
    dcdw2ave=zeros(size(w2));%average
    dcdw1ave=zeros(size(w1));
    dcdk2ave=zeros(size(k2));
    dcdk1ave=zeros(size(k1));
    dcdxc2ave=zeros(size(xc2));
    dcdxc1ave=zeros(size(xc1));
    
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
    clear tmp tmp1
    
    dcdw2ave=dcdw2/m;
    dcdw2ave(:,2:end)=dcdw2ave(:,2:end);%change to zeros(?)
    
    dcdw1ave=dcdw1/m;
    dcdw1ave(:,2:end)=dcdw1ave(:,2:end);
    
    dcdk2ave=sum(dcdk2,2)/m;
    dcdk2ave(:,2:end)=dcdk2ave(:,2:end);
    %the regularization is also problematic, but since it only has one column, the result is correct
    
    dcdk1ave=sum(dcdk1,2)/m;
    dcdk1ave(:,2:end)=dcdk1ave(:,2:end);
    
    dcdxc2ave=sum(dcdxc2,2)/m;
    dcdxc2ave(:,2:end)=dcdxc2ave(:,2:end);
    
    dcdxc1ave=sum(dcdxc1,2)/m;
    dcdxc1ave(:,2:end)=dcdxc1ave(:,2:end);
    
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
    switch modee
        case 1

        case 2
            k2=k2-alpha*dcdk2ave;
            k1=k1-alpha*dcdk1ave;
            xc2=xc2-alpha*dcdxc2ave;
            xc1=xc1-alpha*dcdxc1ave;
        case 3
            k2=k2-alpha*dcdk2ave;
            k1=k1-alpha*dcdk1ave;
        case 4
            %to do
    end
    
    if discretization
        w1 = interp1(roundTheta1,roundTheta1,w1,'nearest');
        w2 = interp1(roundTheta2,roundTheta2,w2,'nearest');
    end
    %find max weight
    if find_max_weight
        [w1_pos_max,w1_neg_max]=find_max_variable(w1,w1_pos_max,w1_neg_max);
        [w2_pos_max,w2_neg_max]=find_max_variable(w2,w2_pos_max,w2_neg_max);
        [k1_pos_max,k1_neg_max]=find_max_variable(k1,k1_pos_max,k1_neg_max);
        [k2_pos_max,k2_neg_max]=find_max_variable(k2,k2_pos_max,k2_neg_max);
        [xc1_pos_max,xc1_neg_max]=find_max_variable(xc1,xc1_pos_max,xc1_neg_max);
        [xc2_pos_max,xc2_neg_max]=find_max_variable(xc2,xc2_pos_max,xc2_neg_max);
    end
    if mod(iter,iter_output)==0
        pred = predict(w1, w2, k1,k2,xc1,xc2,Xtest,firstlayeraf);
        pred_ind=floor(iter/iter_output);
        predic_(pred_ind)=mean(double(pred == ytest)) * 100;
        J_(pred_ind)=sum(sum(-y_tmp.*log(a2)-(1-y_tmp).*log(1-a2)))/m;
    end
end

if find_max_weight
    w1_neg_max=-w1_neg_max;
    w2_neg_max=-w2_neg_max;
    k1_neg_max=-k1_neg_max;
    k2_neg_max=-k2_neg_max;
    xc1_neg_max=-xc1_neg_max;
    xc2_neg_max=-xc2_neg_max;
    w_=[w1_pos_max,w1_neg_max,w2_pos_max,w2_neg_max,k1_pos_max,k1_neg_max,...
        k2_pos_max,k2_neg_max,xc1_pos_max,xc1_neg_max,xc2_pos_max,xc2_neg_max];
else
    w_=0;
end

toc
save('final1.mat','predic_','J_','w_','iterplot');
if (0)
    figure;
    plot(iterplot,predic_,'*')
    figure
    plot(iterplot,J_,'o')
    xlabel('iterations');ylabel('J')
end

