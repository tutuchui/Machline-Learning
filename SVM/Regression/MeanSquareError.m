function MSE = MeanSquareError(Mdl,test_inputs,test_outputs)
dataSize = size(test_inputs,1);
squareError = zeros(dataSize,1);
for i = 1 : dataSize
    predict_labels = predict(Mdl,test_inputs(i,:));
    squareError(i,:) = (predict_labels - test_outputs(i,:))^2;
end
MSE = sum(squareError)/dataSize;

end
