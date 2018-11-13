function tree = decisionTreeLearning(features,labels,validFeatureNo)
examples = [features labels];
featureSize = size(features,2);
% if the all of the examples have the same label, then create a leaf node
% with that label.
if all (~(diff(labels)))
    label = labels(1,1);
    tree = createBinaryTree('Node',[],label,'null','null',label);
% if the size of valid feature is 0, then create a leaf node with the label is
% assigned by the majortiy value of the parent node. 
elseif size(validFeatureNo,2) == 0
    label = MAJORITY_VALUE(labels);
    tree = createBinaryTree('Node',[],label,'null','null',label);
else
    [best_feature,best_threshold] = chooseAttribute(features,labels,validFeatureNo);
% remove the selected feature from validFeature array
    validFeatureNo = validFeatureNo(validFeatureNo ~= best_feature);
    NodeName = ['f',num2str(best_feature),' ',num2str(best_threshold)];
    %
    majorityValue = MAJORITY_VALUE(labels);
    tree = createBinaryTree(NodeName,[],'null',best_feature,best_threshold,majorityValue);
    tree.kids = cell(1,2);
    leftNodeIdx = examples(:,best_feature) >= best_threshold;
    rightNodeIdx = ~leftNodeIdx;
    leftNode = examples(leftNodeIdx,:);
    rightNode = examples(rightNodeIdx,:);
% if the size of example is 0, then create a leaf node with the label is
% assigned by the majortiy value of the parent node. 
    if(size(leftNode,1) == 0)
        label = MAJORITY_VALUE(labels);
        tree.kids{1,1} = createBinaryTree('Node',[],label,'null','null',label);
    else
        subFeatures = leftNode(:,1:featureSize);
        subLabels = leftNode(:,featureSize + 1);
        tree.kids{1,1} = decisionTreeLearning(subFeatures,subLabels,validFeatureNo);
    end
    
    if(size(rightNode,1) == 0)
        label = MAJORITY_VALUE(labels);
        tree.kids{1,2} = createBinaryTree('Node',[],label,'null','null',label);
    else     
        subFeatures = rightNode(:,1:featureSize);
        subLabels = rightNode(:,featureSize + 1);
        tree.kids{1,2} = decisionTreeLearning(subFeatures,subLabels,validFeatureNo);
    end 
end
end