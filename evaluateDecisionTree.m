function [cmat,recall,precision,fscore] = evaluateDecisionTree(test_inputs,test_targets,tree)
   outputs = zeros(15,1);
   for n = 1:size(test_inputs,1)
        sample = test_inputs(n,:);
        class = validTree(tree,sample);
        outputs(n,1) = class;
   end
    cmat = ConfusionMatrix(test_targets, outputs(:,1));
    TN = cmat(1,1); FP = cmat(1,2); TP = cmat(2,2); FN = cat(2,1);
    recall = TP/(TP+FN);
    precision = TP/(TP+FP);
    fscore = (2*precision * recall)/(precision +recall);
end