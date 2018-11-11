%confusion matrix function
function confusion_matrix = ConfusionMatrix (targets, outputs)
    values = max(targets) - min(targets) + 1;
    confusion_matrix = zeros(values);
    for i = 1:length(targets)
        row = targets(i,1)+1;
        col = outputs(i,1)+1;
        confusion_matrix(row,col) =  confusion_matrix(row,col) + 1;
    end
end
