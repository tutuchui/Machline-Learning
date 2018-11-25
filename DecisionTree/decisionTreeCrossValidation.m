%cross validation of descision tree
load('facialPoints.mat');
load('labels.mat');
inputs = reshape(points,132,150);
inputs = inputs';
targets = labels;
[inputs,targets] = shuffleMatrix(inputs,targets);
%generate boolean indicies for 10-fold cross validation
indices = crossvalind('Kfold', targets,10);
recallSet = zeros(10,1);
precisionSet = zeros(10, 1);
fscoreSet = zeros(10, 1);
accuracySet = zeros(10,1);
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(test_set,:);
    train_inputs = inputs(train_set,:);
    test_targets = targets(test_set,:);
    train_targets = targets(train_set,:);
    %train the tree
    validFeatureNo = 1 : size(train_inputs,2);
    tree = decisionTreeLearning(train_inputs, train_targets,validFeatureNo);
    
    %feed the test data through the tree
    [cmat,recall,precision,fscore,missclassifiedNode,accuracy] = evaluateDecisionTree(test_inputs,test_targets,tree);
    accuracySet(i,1) = accuracy;
    %create confusion matrix and fscorei
    recallSet(i,1) = recall;
    precisionSet(i,1) = precision;
    fscoreSet(i,1) = fscore;
end
