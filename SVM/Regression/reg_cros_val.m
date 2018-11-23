load('facialPoints.mat');
load('labels.mat');
labels = pose(:,6);
inputs = transpose(reshape(points,132,8955));
[inputs, labels] = shuffleMatrix(inputs, labels);
indicies = zeros(length(inputs), 1);
index = 1;
RMSE = zeros(10, 1);
%  classificationRate = zeros(10, 1);

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
    Mdl = fitrsvm(train_inputs,train_targets,'Kernal function', 'linear','KernalScale',best_sigma,'BoxConstraint', best_c);
    
    outputs = predict(Mdl, test_inputs);
    
    RMSE(i,1) = sqrt((1/2*(length(test_targets))*(outputs - test_targets)^2));  
    
end