%cross validation of descision tree
points = load("facialPoints.mat");
targets = load("labels.mat");
inputs = transpose(reshape(points,132,150));

%generate boolean indicies for 10-fold cross validation
indices = crossvalind('Kfold', targets,10);
recall = zeros(10,1);
precision = zeros(10, 1);
fscore = zeros(10, 1);
for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(:,test_set);
    train_inputs = inputs(:,train_set);
    test_targets = targets(:,test_set);
    train_targets = targets(:,train_set);
    %train the tree
    tree = decision_tree_learning(train_inputs, train_targets);
    
    %feed the test data through the tree
    outputs = zeros([length(test_inputs) 1]);
    for n = 1:size(test_inputs)
        x = test_inputs(:,1);
        while tree.op ~= "leaf_node"
            attr_val = x(tree.attribute);
            if attr_val < tree.threshold
                tree = tree(1).kids;
            else
                tree = tree(2).kids;
            end
        end
        outputs(n) = (tree.class);
    end
    
    %create confusion matrix and fscore
    cmat = ConfusionMatrix(targets, outputs);
    TP = cmat(1,1); FP = cmat(1,2); TN = cmat(2,2); FN = cat(2,1);
    recall(i) = TP/(TP+FN);
    precision(i) = TP/(TP+FP);
    fscore(i) = (2*precision(i)*recall(i))/(precision(i)+recall(i));
end
