function optimalParaValue = RegressionInnerCrossVal(inputs,labels,kernelName,parameterName)
indices = crossvalind('Kfold',labels,3);

% Set a search range for the BoxConstraint and Epsilon
% The range of BoxConstraint is 2^-5 to 2^10, ratio is root 2.
% The range of Epsilon is 0.1 to 2, step is 0.1.
GridC = 2.^(-5:1:10);
GridEpsilon = 0.1 : 0.2 : 2;
GridSigma = 1e-3 : 2 : 30;
GridPoly = 1 : 1 : 6;
% if the kernelFunction is RBF.
if(strcmp(kernelName,'RBF'))
    minLoss = inf;
    if(strcmp(parameterName,'BoxConstraint'))
        for C = GridC
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','BoxConstraint',C);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = C;
                disp(['C:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    elseif(strcmp(parameterName,'KernelScale'))
        for sigma = GridSigma
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','KernelScale',sigma);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = sigma;
                disp(['sigma:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    elseif(strcmp(parameterName,'Epsilon'))
        for epsilon = GridEpsilon
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','Epsilon',epsilon);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = epsilon;
                disp(['epsilon:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    end
%if the kernel is polynomial
elseif(strcmp(kernelName,'polynomial'))
    minLoss = inf;
    if(strcmp(parameterName,'BoxConstraint'))
        for C = GridC
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','polynomial','BoxConstraint',C);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = C;
                disp(['C:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    elseif(strcmp(parameterName,'KernelScale'))
        for q = GridPoly
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','polynomial','PolynomialOrder',q);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = q;
                disp(['sigma:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    elseif(strcmp(parameterName,'Epsilon'))
        for epsilon = GridEpsilon
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','polynomial','Epsilon',epsilon);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = epsilon;
                disp(['epsilon:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    end

else
    exception = MException('%s is not a valid kernel name for this function',kernelName);
    throw(exception);
end
end