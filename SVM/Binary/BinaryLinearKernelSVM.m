load('facialPoints.mat');
load('labels.mat');

points = reshape(points,132,150);
X = points';
Y = labels;
Mdl = fitcsvm(X,Y,'KernelFunction','linear','BoxConstraint',1);

indices = crossvalind('Kfold',Y,10);
foldSize = size(X,1) / 10;
predict_Y = zeros(15,10);
accuracy = zeros(10,1);
for i = 1 : 10
    test_set = (indices == i);
    train_set = ~test_set;
    test_X = X(test_set,:);
    test_Y = Y(test_set,:);
    train_X = X(train_set,:);
    train_Y = Y(train_set,:);
    
    Mdl = fitcsvm(train_X,train_Y,'KernelFunction','linear','BoxConstraint',1);
    for j = 1 : foldSize
        [predict_Y(j,i),score] = predict(Mdl,test_X(j,:));
    end
   
    accuracy(i,1) = (foldSize - sum(xor(predict_Y(:,i),test_Y))) / foldSize;
end

