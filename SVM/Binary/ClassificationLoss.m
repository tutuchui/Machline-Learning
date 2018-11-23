function Loss = ClassificationLoss(Mdl,test_inputs,test_labels)
dataSize = size(test_inputs,1);
    predict_labels = zeros(dataSize,1);
    for i = 1 : dataSize
        [predict_labels(i,:),score] = predict(Mdl,test_inputs(i,:));
    end
    
    Loss = sum(xor(predict_labels(:,i),test_labels)) / dataSize;
end