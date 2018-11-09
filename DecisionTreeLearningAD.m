function decisionTree = DecisionTreeLearning(features, labels)
examples = [features labels];
if all(examples(:,2) == examples(1,2))
    decisionTree = tree(examples(1,:));
else
    best = ChooseAttribute(features,labels);
    decisionTree = tree(best(1));
    for i = 1 : length(examples)
        if examples(i,1) == best(2)
            examples(i,:) = [];
        elseif examples(i,1) > best(2)
            examplesLeft = [examplesLeft;examples(i,:)];
        else
            examplesRight = [examplesRight;examples(i,:)];
        end
    end
    examplesi = [examplesLeft examplesRight];
    for i = 1 : length(examplesi)
        if isempty(examplesi(i)) == 1
            decisionTree = tree(MAJORITY_VALUE(best(1)));
        else
            newExamples = examplesi(i);
            featuresi = newExamples(:,1);
            labelsi = newExamples(:,2);
            decisionTreei = DecisionTreeLearning(featuresi, labelsi);
            decisionTree = decisionTree.addNode(1,decisionTreei);
        end
    end    
end