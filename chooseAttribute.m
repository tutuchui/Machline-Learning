function [best_feature,best_threshold] = chooseAttribute(features,targets)
    examples = [features targets];
    maxGainInfo = -100000;
    for i = 1 : size(features,2)
       thresholds  = features(:,i);
       for j = 1 : size(thresholds,1)
           threshold = round(thresholds(j,1)) - 1;
           gainInfo = calculateGainInfo(i,examples,threshold);
           if gainInfo > maxGainInfo
               best_feature = i;
               best_threshold = threshold;
               maxGainInfo = gainInfo;
           end
       end
    end
end