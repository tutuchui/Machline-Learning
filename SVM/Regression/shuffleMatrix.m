function [shuffle_inputs,shuffle_outputs] =  shuffleMatrix(inputs,label)
    inputs_labels = [inputs label];
    shuffle_matrix = inputs_labels(randperm(size(inputs_labels,1)),:);
    shuffle_inputs = shuffle_matrix(:,1:size(inputs,2));
    shuffle_outputs = shuffle_matrix(:,size(shuffle_matrix,2));
end
