load('facialPoints.mat');
load('labels.mat');
points = reshape(points,132,150);
features = points';
targets = labels;
validFeatureNo = 1 : size(features,2);
decisionTree = decisionTreeLearning(features,targets,validFeatureNo);