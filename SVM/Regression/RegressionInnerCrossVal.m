function optimalParaValue = RegressionInnerCrossVal(inputs,labels,kernelName,parameterName)
indices = crossvalind('Kfold',labels,3);

% Set a search range for the BoxConstraint and Epsilon
% The range of BoxConstraint is 1e-3 to 1000, step 50.
% The range of Epsilon is 0.1 to 2, step is 0.2.
% The range of polynomial order is 1 to 10 and step is 1;
% The range of kernel scale sigma is 1e-3 to 100, step is 10;
GridC = 1e-3 : 100 : 500 + 1e-3;
GridEpsilon = 0.1 : 0.2 : 2;
GridSigma = 1e-3 : 10 : 100;
GridPoly = 1 : 1 : 10;
% if the kernelFunction is RBF.
if(strcmp(kernelName,'RBF'))
    minLoss = inf;
    if(strcmp(parameterName,'BoxConstraint'))
        %Do the grid search for BoxConstraint
        for C = GridC
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','BoxConstraint',C,'KernelScale',40);
                % Calculate the root mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            disp(['C:',num2str(C),' MSE:',num2str(aveMSE)]);
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = C;
            end
        end
        for C = optimalParaValue - 20 : 5 : optimalParaValue + 20
             totalMSE = 0;
             for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','BoxConstraint',C,'KernelScale',40);
                % Calculate the root mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            disp(['C:',num2str(C),' MSE:',num2str(aveMSE)]);
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = C;
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
                % Calculate the root mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
             disp(['sigma:',num2str(sigma),' MSE:',num2str(aveMSE)]);
            %If the average root mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = sigma;
            end
        end
        for sigma = optimalParaValue - 5 : optimalParaValue + 5
            totalMSE = 0;
            for i = 1 : 3
                test_set = (indices == i);
                train_set = ~test_set;
                test_inputs = inputs(test_set,:);
                test_labels = labels(test_set,:);
                train_inputs = inputs(train_set,:);
                train_labels = labels(train_set,:);
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','KernelScale',sigma);
                % Calculate the root mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
             disp(['sigma:',num2str(sigma),' MSE:',num2str(aveMSE)]);
            %If the average root mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = sigma;
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
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','Epsilon',epsilon,'KernelScale',40);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            disp(['epsilon:',num2str(epsilon),' MSE:',num2str(aveMSE)]);
            %If the average mean squre error less than current minimal
            %mean squre error.
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = epsilon;
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
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','polynomial','BoxConstraint',C,'PolynomialOrder',1);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            disp(['C:',num2str(C),' MSE:',num2str(aveMSE)]);
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = C;
%                 disp(['C:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
            end
        end
    elseif(strcmp(parameterName,'PolynomialOrder'))
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
            disp(['PolynomialOrder:',num2str(q),' MSE:',num2str(aveMSE)]);
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = q;
                disp(['PolynomialOrder:',num2str(optimalParaValue),' MSE:',num2str(minLoss)]);
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
                svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','polynomial','Epsilon',epsilon,'PolynomialOrder',1);
                % Calculate the mean square error for the model.
                MSE = MeanSquareError(svmMdl,test_inputs,test_labels);
                totalMSE = totalMSE + MSE;
            end
            aveMSE = totalMSE / 3;
            %If the average mean squre error less than current minimal
            %mean squre error.
            disp(['Epsilon:',num2str(epsilon),' MSE:',num2str(aveMSE)]);
            if aveMSE < minLoss
                minLoss = aveMSE;
                optimalParaValue = epsilon;
            end
        end
    end

else
    exception = MException('%s is not a valid kernel name for this function',kernelName);
    throw(exception);
end
end