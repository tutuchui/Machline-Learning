%loading and transforming the data
load('facialPoints.mat');
load('labels.mat');
points = reshape(points,132,150);
features = points';
targets = labels;

%Initial the valid feature set
validFeatureNo = 1 : size(features,2);
decisionTree = decisionTreeLearning(features,targets,validFeatureNo);