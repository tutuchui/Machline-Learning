load('facialPoints.mat');
load('headpose.mat');
points = reshape(points,132,8955);
X = points';
Y = pose(:,6);

Mdl = fitrsvm(X,Y,'kernelFunction','linear','Epsilon',0.2,'BoxConstraint',1);
