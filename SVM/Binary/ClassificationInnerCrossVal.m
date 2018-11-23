function optimalParaValue = ClassificationInnerCrossVal(inputs,labels,kernelName,parameterName)
    %Split the data into 10 fold. 
    indices = crossvalind('Kfold',labels,3);
    % Set a search range for the BoxConstraint
    % The range of BoxConstraint is 2^-5 to 2^10, ratio is root 2.
    GridC = 2.^(-5:0.5:10);
    GridSigma = 1e-3 : 0.1 : 30;
    GridPoly = 1 : 1 : 10;
    % if the kernelFunction is RBF. 
    if(strcmp(kernelName,'RBF'))
        minLoss = inf;
        if(strcmp(parameterName,'BoxConstraint'))
            for C = GridC
                totalLoss = 0;
                for i = 1 : 3
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','RBF','BoxConstraint',C);
                    %Calculate the classification loss for the trained
                    %model
                    L = ClassificationLoss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 3;
                if aveLoss < minLoss
                        minLoss = aveLoss;
                        optimalParaValue = C;
                        disp(['C:',num2str(optimalParaValue),' Loss:',num2str(minLoss)]);
                end
            end   
        elseif(strcmp(parameterName,'KernelScale'))
            for sigma = GridSigma
                totalLoss = 0;
                for i = 1 : 3
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','RBF','KernelScale',sigma);
                    %Calculate the classification loss for the trained
                    %model
                    L = ClassificationLoss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 3;
                if aveLoss < minLoss
                        minLoss = aveLoss;
                        optimalParaValue = sigma;
                        disp(['Sigma:',num2str(optimalParaValue),' Loss:',num2str(minLoss)]);
                end
            end
        end 
        
    elseif(strcmp(kernelName,'polynomial'))
        minLoss = inf;
        if(strcmp(parameterName,'BoxConstraint'))
            for C = GridC
                totalLoss = 0;
                for i = 1 : 3
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','polynomial','BoxConstraint',C);
                    %Calculate the classification loss for the trained
                    %model
                    L = ClassificationLoss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 3;
                if aveLoss < minLoss
                        minLoss = aveLoss;
                        optimalParaValue = C;
                        disp(['C:',num2str(optimalParaValue),' Loss:',num2str(minLoss)]);
                end
            end   
        elseif(strcmp(parameterName,'PolynomialOrder'))
            for g = GridPoly
                totalLoss = 0;
                for i = 1 : 3
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','polynomial','PolynomialOrder',g);
                    %Calculate the classification loss for the trained
                    %model
                    L = ClassificationLoss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 3;
                if aveLoss < minLoss
                        minLoss = aveLoss;
                        optimalParaValue = g;
                        disp(['Order:',num2str(optimalParaValue),' Loss:',num2str(minLoss)]);
                end
            end
        end 
    else
        exception = MException('%s is not a valid kernel name for this function',kernelName);
        throw(exception);
    end
end