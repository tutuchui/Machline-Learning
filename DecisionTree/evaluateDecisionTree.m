function [cmat,recall,precision,fscore,missclassifiedNode,accuarcy] = evaluateDecisionTree(test_inputs,test_targets,tree)
   outputs = zeros(size(test_targets,1),1);
   j = 1;
   missclassifiedNode = zeros(15,132);
   for n = 1:size(test_inputs,1)
        sample = test_inputs(n,:);
        class = validTree(tree,sample);
        outputs(n,1) = class;
        if class ~= test_targets(n,:)
            missclassifiedNode(j,:) = sample;
            j = j + 1;
        end
   end
    cmat = ConfusionMatrix(test_targets, outputs(:,1));
    accuarcy = 1 - (j-1)/size(test_targets,1);
    TN = cmat(1,1); FP = cmat(1,2); TP = cmat(2,2); FN = cat(2,1);
    recall = TP/(TP+FN);
    precision = TP/(TP+FP);
    fscore = (2*precision * recall)/(precision +recall);
end