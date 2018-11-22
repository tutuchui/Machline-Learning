function [bestC,bestSigma,bestQ,bestEpsilon] = RegressionInnerCrossVal(inputs,labels,kernelName)
indices = crossvalind('Kfold',labels,10);

% Set a search range for the BoxConstraint and Epsilon
% The range of BoxConstraint is 2^-5 to 2^10, ratio is root 2.
% The range of Epsilon is 0.1 to 2, step is 0.1.
GridC = 2.^(-5:0.5:10);
GridEpsilon = 0.1 : 0.1 : 3;
% if the kernelFunction is RBF.
if(strcmp(kernelName,'RBF'))
    GridSigma = 1e-3 : 0.1 : 30;
    minLoss = inf;
    for epsilon = GridEpsilon
        for sigma = GridSigma
            for C = GridC
                totalRSE = 0;
                for i = 1 : 10
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitrsvm(train_inputs,train_labels,'kernelFunction','RBF','Epsilon',epsilon,'BoxConstraint',C,'KernelScale',sigma);
                    % Calculate the mean square error for the model.
                    RSE = loss(svmMdl,test_inputs,test_labels);
                    totalRSE = totalRSE + RSE;
                end
                aveRSE = totalRSE / 10;
                %If the average mean squre error less than current minimal
                %mean squre error.
                if aveRSE < minLoss
                    minLoss = aveRSE;
                    bestC = C;
                    bestSigma = sigma;
                    bestEpsilon = epsilon;
                    disp(['C:',num2str(bestC),' sigma:',num2str(bestSigma)]);
                end
            end
        end
        bestQ = 'null';
    end
%if the kernel is polynomial
elseif(strcmp(kernelName,'polynomial'))
    minLoss = inf;
%Set the for the polynomial order, which is 1 to 10, step is 1.
    GridPoly = 1 : 1 : 10;
    for epsilon = GridEpsilon
        for q = GridPoly
            for C = GridC
                totalRSE = 0;
                for i = 1 : 10
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitrsvm(train_inputs,train_labels,'KernelFunction','polynomial','BoxConstraint',C,'PolynomialOrder',q,'Epsilon',epsilon);
                    RSE = loss(svmMdl,test_inputs,test_labels);
                    totalRSE = totalRSE + RSE;
                end
                aveRSE = totalRSE / 10;
                if aveRSE < minLoss
                    minLoss = aveRSE;
                    bestC = C;
                    bestQ = q;
                    bestEpsilon = epsilon;
                    
                end
            end
        end
    end
    bestSigma = 'null';
    
else
    exception = MException('%s is not a valid kernel name for this function',kernelName);
    throw(exception);
end
end