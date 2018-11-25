load('facialPoints.mat');
load('labels.mat');
inputs = transpose(reshape(points,132,150));
[inputs, labels] = shuffleMatrix(inputs, labels);
indices = crossValIndices(labels,10);
% error = zeros(10, 1);
LinerClassificationRate = zeros(10, 1);
RBFClassificationRate = zeros(10,1);
PolyClassificationRate = zeros(10,1);
%
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

for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);   
    Mdl = fitcsvm(train_inputs,train_targets,'Kernelfunction', 'RBF','BoxConstraint',100.001,'KernelScale',30.001);
    accuracy = 1 - ClassificationLoss(Mdl,test_inputs,test_targets);
    RBFClassificationRate(i,1) = accuracy;        
end

for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = labels(test_set,:);
    train_targets = labels(train_set,:);   
    Mdl = fitcsvm(train_inputs,train_targets,'Kernelfunction', 'polynomial','BoxConstraint',0.001,'PolynomialOrder',1);
    accuracy = 1 - ClassificationLoss(Mdl,test_inputs,test_targets);
    PolyClassificationRate(i,1) = accuracy;        
end
