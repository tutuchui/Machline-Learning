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
%  classificationRate = zeros(10, 1);
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

for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    
    %train the SVM using innerfold-cross validation
    Mdl = fitrsvm(train_inputs,train_targets,'Kernelfunction', 'RBF','Epsilon',0.3,'BoxConstraint', 100.001,'KernelScale',56.001);
    
    RbfRMSE(i,1) = MeanSquareError(Mdl,test_inputs,test_targets);  
end

for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);
    
    %train the SVM using innerfold-cross validation
    Mdl = fitrsvm(train_inputs,train_targets,'Kernelfunction', 'polynomial','Epsilon',1.9,'BoxConstraint', 0.001,'PolynomialOrder',1);   
    PolyRMSE(i,1) = MeanSquareError(Mdl,test_inputs,test_targets);  
end