load('facialPoints.mat');
load('labels.mat');

points = reshape(points,132,150);
X = points';
Y = labels;
%Trian the model using Gussian kernel
% [bestC_rbf,bestSigma,~] = ClassificationInnerCrossVal(X,Y,'RBF');
% rbfModel = fitcsvm(X,Y,'KernelFunction','RBF','BoxConstraint',bestC_rbf,'KernelScale',bestSigma);
% rbf_a = size(rbfModel.SupportVectors,1) / size(Y,1);

[bestC_poly,~,bestQ] = ClassificationInnerCrossVal(X,Y,'polynomial');
polyModel = fitcsvm(X,Y,'KernelFunction','polynomial','BoxConstraint',bestC_poly,'PolynomialOrder',bestQ);
poly_a = size(polyModel.SupportVectors,1) / size(Y,1);
