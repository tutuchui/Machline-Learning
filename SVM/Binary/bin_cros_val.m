load('facialPoints.mat');
load('labels.mat');
inputs = transpose(reshape(points,132,150));
[inputs, labels] = shuffleMatrix(inputs, labels);
indicies = zeros(length(inputs), 1);
index = 1;
% error = zeros(10, 1);
classificationRate = zeros(10, 1);

for i=1:length(inputs)
    indicies(i, 1) = index;
    index = index + 1;
    if (index == 11)
        index = 1;
    end
end

for i =1:10
    %select training and test sets for crossvalidation
    test_set = (indices == i);
    train_set = ~test_set;
    test_inputs = inputs(:,test_set);
    train_inputs = inputs(:,train_set);
    test_targets = labels(:,test_set);
    train_targets = labels(:,train_set);
    
    %train the SVM using innerfold-cross validation
    Mdl = fitcsvm(train_inputs,train_targets,'Kernal function', 'linear','KernalScale',best_sigma,'BoxConstraint', best_c);
    
    outputs = predict(Mdl, test_inputs);
    
%     error(i,1) = (1/2*(length(test_targets))*(outputs - test_targets)^2;
    correct = 0;
    for j = 1:length(test_inputs)
        if (test_targets(j,1) == outputs(j,1))
            correct = correct + 1;
        end
    end
    
    classificationRate(i,1) = correct/(length(test_targets));    
    
end