function [bestC,bestSigma,bestQ] = ClassificationInnerCrossVal(inputs,labels,kernelName)
    %Split the data into 10 fold. 
    indices = crossvalind('Kfold',labels,10);
    % Set a search range for the BoxConstraint
    % The range of BoxConstraint is 2^-5 to 2^10, ratio is root 2.
    GridC = 2.^(-5:0.5:10);
    % if the kernelFunction is RBF. 
    if(strcmp(kernelName,'RBF'))
        GridSigma = 1e-3 : 0.1 : 30;
        minLoss = inf;
        for sigma = GridSigma
            for C = GridC
                totalLoss = 0;
                for i = 1 : 10 
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','RBF','BoxConstraint',C,'KernelScale',sigma);
                    %Calculate the classification loss for the trained
                    %model
                    L = loss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 10;
                if aveLoss < minLoss
                        minLoss = L;
                        bestC = C;
                        bestSigma = sigma;
                end
            end
            bestQ = 'null';
        end
    elseif(strcmp(kernelName,'polynomial'))
        minLoss = inf;
        GridPoly = 1 : 1 : 10;
        for q = GridPoly
            for C = GridC
                totalLoss = 0;
                for i = 1 : 10  
                    test_set = (indices == i);
                    train_set = ~test_set;
                    test_inputs = inputs(test_set,:);
                    test_labels = labels(test_set,:);
                    train_inputs = inputs(train_set,:);
                    train_labels = labels(train_set,:);
                    svmMdl = fitcsvm(train_inputs,train_labels,'KernelFunction','polynomial','BoxConstraint',C,'PolynomialOrder',q);
                    %Calculate the classification loss for the trained
                    %model
                    L = loss(svmMdl,test_inputs,test_labels);
                    totalLoss = totalLoss + L;
                end
                aveLoss = totalLoss / 10;
                if aveLoss < minLoss
                    minLoss = L;
                    bestC = C;
                    bestQ = q;
                end
            end
        end
        bestSigma = 'null';
    else
        exception = MException('%s is not a valid kernel name for this function',kernelName);
        throw(exception);
    end
end