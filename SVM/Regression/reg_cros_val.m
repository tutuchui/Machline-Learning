load('facialPoints.mat');
load('headpose.mat');
labels = pose(:,6);
inputs = reshape(points,132,8955);
inputs = inputs';
[inputs, labels] = shuffleMatrix(inputs, labels);
indices = crossValIndices(labels,10);
LinerRMSE = zeros(10, 1);
RbfRMSE = zeros(10, 1);
PolyRMSE = zeros(10, 1);
classificationRate = zeros(10, 1);
bestC_rbfs = zeros(10,1);
bestSigmas = zeros(10,1);
bestEpsilons_rbf = zeros(10,1);
bestC_polys = zeros(10,1);
bestOrders = zeros(10,1);
bestEpsilons_poly = zeros(10,1);
%% Cross Validation for linear kernel
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    %train the SVM using innerfold-cross validation
    Mdl = fitrsvm(train_inputs,train_targets,'Kernelfunction', 'linear','Epsilon',1.9,'BoxConstraint', 0.001);
    
    LinerRMSE(i,1) = MeanSquareError(Mdl,test_inputs,test_targets);  
end
%% Cross Validation for rbf kernel 
for i =1:10
    select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    disp(['time: ',num2str(i)]);
    bestC_rbf = RegressionInnerCrossVal(train_inputs,train_targets,'RBF','BoxConstraint');
    bestSigma = RegressionInnerCrossVal(train_inputs,train_targets,'RBF','KernelScale');
    bestEpsilon_rbf = RegressionInnerCrossVal(train_inputs,train_targets,'RBF','Epsilon');
    bestC_rbfs(i,:) = bestC_rbf;
    bestSigmas(i,:) = bestSigma;
    bestEpsilons_rbf(i,:) = bestEpsilons_rbf;
    %train the SVM using innerfold-cross validation
    Mdl = fitrsvm(train_inputs,train_targets,'Kernelfunction', 'RBF','Epsilon',bestEpsilon_rbf,'BoxConstraint', bestC_rbf,'KernelScale',bestSigma);
    rbf_a(i,:) = size(Mdl.SupportVectors,1) / size(train_inputs,1);
    RbfRMSE(i,1) = MeanSquareError(Mdl,test_inputs,test_targets);  
end
%% Cross Validation for polynomial kernel 
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    %train the SVM using innerfold-cross validation
    disp(['time: ',num2str(i)]);
    bestC_poly = RegressionInnerCrossVal(train_inputs,train_targets,'polynomial','BoxConstraint');
    bestQ = RegressionInnerCrossVal(train_inputs,train_targets,'polynomial','PolynomialOrder');
    bestEpsilon_poly = RegressionInnerCrossVal(train_inputs,train_targets,'polynomial','Epsilon');
    bestC_polys(i,:) = bestC_poly;
    bestOrders(i,:) = bestQ;
    bestEpsilons_poly(i,:) = bestEpsilon_poly;
    Mdl = fitrsvm(train_inputs,train_targets,'Kernelfunction', 'polynomial','Epsilon',bestEpsilon_poly,'BoxConstraint', bestC_poly,'PolynomialOrder',bestQ);
    poly_a(i,:) = size(Mdl.SupportVectors,1) / size(train_inputs,1);
    PolyRMSE(i,1) = MeanSquareError(Mdl,test_inputs,test_targets);  
end