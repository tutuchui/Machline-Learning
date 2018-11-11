%cross validation of descision tree
load('facialPoints.mat');
load('labels.mat');
inputs = reshape(points,132,150);
inputs = inputs';
targets = labels;
[inputs,targets] = shuffleMatrix(inputs,targets);
%generate boolean indicies for 10-fold cross validation
indices = crossvalind('Kfold', targets,10);
recall = zeros(10,1);
precision = zeros(10, 1);
fscore = zeros(10, 1);
outputs = zeros(15,10);
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
   
    for n = 1:size(test_inputs,1)
        sample = test_inputs(n,:);
        class = validTree(tree,sample);
        outputs(n,i) = class;
    end
    
    %create confusion matrix and fscorei
    cmat = ConfusionMatrix(test_targets, outputs(:,i));
    TN = cmat(1,1); FP = cmat(1,2); TP = cmat(2,2); FN = cat(2,1);
    recall(i,1) = TP/(TP+FN);
    precision(i,1) = TP/(TP+FP);
    fscore(i,1) = (2*precision(i,1)*recall(i,1))/(precision(i,1)+recall(i,1));
end
