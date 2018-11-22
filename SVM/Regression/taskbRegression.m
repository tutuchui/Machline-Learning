load('facialPoints.mat');
load('headpose.mat');
points = reshape(points,132,8955);
X = points';
Y = pose(:,6);

%Trian the model using Gussian kernel
[bestC,bestSigma,~,bestEpsilon] = RegressionInnerCrossVal(X,Y,'RBF');
rbfModel = fitrsvm(train_inputs,train_labels,'KernelFunction','RBF','BoxConstraint',bestC,'KernelScale',bestSigma,'epsilon',bestEpsilon);
rbf_a = size(rbfModel.supportVectors,1) / size(Y,1);

[bestC,~,bestQ,bestEpsilon] = ClassificationInnerCrossVal(X,Y,'polynomial');
polyModel = fitrsvm(train_inputs,train_labels,'KernelFunction','polynomial','BoxConstraint',bestC,'PolynomialOrder',bestQ,'epsilon',bestEpsilon);
poly_a = size(polyModel.supportVectors,1) / size(Y,1);
