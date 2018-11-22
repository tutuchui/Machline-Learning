load('facialPoints.mat');
load('labels.mat');

points = reshape(points,132,150);
X = points';
Y = labels;
%Trian the model using Gussian kernel
[bestC,bestSigma,~] = ClassificationInnerCrossVal(X,Y,'RBF');
rbfModel = fitcsvm(train_inputs,train_labels,'KernelFunction','RBF','BoxConstraint',bestC,'KernelScale',bestSigma);
rbf_a = size(rbfModel.supportVectors,1) / size(Y,1);

[bestC,~,bestQ] = ClassificationInnerCrossVal(X,Y,'polynomial');
polyModel = fitcsvm(train_inputs,train_labels,'KernelFunction','polynomial','BoxConstraint',bestC,'PolynomialOrder',bestQ);
poly_a = size(polyModel.supportVectors,1) / size(Y,1);
