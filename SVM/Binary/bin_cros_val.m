%% Data loading and transforming
load('facialPoints.mat');
load('labels.mat');
inputs = transpose(reshape(points,132,150));
[inputs, labels] = shuffleMatrix(inputs, labels);
indices = crossValIndices(labels,10);
% error = zeros(10, 1);
LinerClassificationRate = zeros(10, 1);
RBFClassificationRate = zeros(10,1);
PolyClassificationRate = zeros(10,1);
bestC_rbfs = zeros(10,1);
bestSigmas = zeros(10,1);
bestC_polys = zeros(10,1);
bestOrders = zeros(10,1);
%% Cross Validation for linera function
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);   
    %train the SVM using innerfold-cross validation
    Mdl = fitcsvm(train_inputs,train_targets,'Kernelfunction', 'linear','BoxConstraint',1);
    accuracy = 1 - ClassificationLoss(Mdl,test_inputs,test_targets);
    LinerClassificationRate(i,1) = accuracy;        
end
%% Cross Validation for rbf kernel 
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    bestC_rbf = ClassificationInnerCrossVal(train_inputs,train_targets,'RBF','BoxConstraint');
    bestSigma = ClassificationInnerCrossVal(train_inputs,train_targets,'RBF','KernelScale');
    bestC_rbfs(i,:) = bestC_rbf;
    bestSigmas(i,:) = bestSigma;
    Mdl = fitcsvm(train_inputs,train_targets,'Kernelfunction', 'RBF','BoxConstraint',bestC_rbf,'KernelScale',bestSigma);
    rbf_a(i,:) = size(Mdl.SupportVectors,1) / size(train_inputs,1);
    accuracy = 1 - ClassificationLoss(Mdl,test_inputs,test_targets);
    RBFClassificationRate(i,1) = accuracy;        
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
    bestC_poly = ClassificationInnerCrossVal(train_inputs,train_targets,'polynomial','BoxConstraint');
    bestPoly = ClassificationInnerCrossVal(train_inputs,train_targets,'polynomial','PolynomialOrder');
    bestC_polys(i,:) = bestC_poly;
    bestOrders = bestPoly;
    Mdl = fitcsvm(train_inputs,train_targets,'Kernelfunction', 'polynomial','BoxConstraint',bestC_poly,'PolynomialOrder',bestPoly);
    rbf_poly(i,:) = size(Mdl.SupportVectors,1) / size(train_inputs,1);
    accuracy = 1 - ClassificationLoss(Mdl,test_inputs,test_targets);
    PolyClassificationRate(i,1) = accuracy;        
end
