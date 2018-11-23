function Loss = ClassificationLoss(Mdl,test_inputs,test_labels)
    dataSize = size(test_inputs,1);
    predict_labels = predict(Mdl,test_inputs);
    Loss = sum(xor(predict_labels,test_labels)) / dataSize;
end