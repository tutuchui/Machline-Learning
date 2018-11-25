load('facialPoints.mat');
load('headpose.mat');
points = reshape(points,132,8955);
X = points';
Y = pose(:,6);

% Trian the model using Gussian kernel
bestC_rbf = RegressionInnerCrossVal(X,Y,'RBF','BoxConstraint');
bestSigma = RegressionInnerCrossVal(X,Y,'RBF','KernelScale');
bestEpsilon_rbf = RegressionInnerCrossVal(X,Y,'RBF','Epsilon');

rbfModel = fitrsvm(X,Y,'KernelFunction','RBF','BoxConstraint',bestC_rbf,'KernelScale',bestSigma,'epsilon',bestEpsilon_rbf);
rbf_a = size(rbfModel.SupportVectors,1) / size(Y,1);

% Trian the model using polynomial kernel
bestC_poly = RegressionInnerCrossVal(X,Y,'polynomial','BoxConstraint');
bestQ = RegressionInnerCrossVal(X,Y,'polynomial','PolynomialOrder');
bestEpsilon_poly = RegressionInnerCrossVal(X,Y,'polynomial','Epsilon');
polyModel = fitrsvm(X,Y,'KernelFunction','polynomial','BoxConstraint',bestC_poly,'PolynomialOrder',bestQ,'epsilon',bestEpsilon_poly);
poly_a = size(polyModel.SupportVectors,1) / size(Y,1);
