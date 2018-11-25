load('facialPoints.mat');
load('labels.mat');

points = reshape(points,132,150);
X = points';
Y = labels;
%Trian the model using Gussian kernel
bestC_rbf = ClassificationInnerCrossVal(X,Y,'RBF','BoxConstraint');
bestSigma = ClassificationInnerCrossVal(X,Y,'RBF','KernelScale');
rbfModel = fitcsvm(X,Y,'KernelFunction','RBF','BoxConstraint',bestC_rbf,'KernelScale',bestSigma);
% rbf_a = size(rbfModel.SupportVectors,1) / size(Y,1);

% [bestC_poly,~,bestQ] = ClassificationInnerCrossVal(X,Y,'polynomial');
bestC_poly = ClassificationInnerCrossVal(X,Y,'polynomial','BoxConstraint');
bestPoly = ClassificationInnerCrossVal(X,Y,'polynomial','PolynomialOrder');
polyModel = fitcsvm(X,Y,'KernelFunction','polynomial','BoxConstraint',bestC_poly,'PolynomialOrder',bestPoly);
% poly_a = size(polyModel.SupportVectors,1) / size(Y,1);
