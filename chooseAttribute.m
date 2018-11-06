function [best_feature,best_threshold] = chooseAttribute(features,targets,validFeatureNo)
    examples = [features targets];
    maxGainInfo = -inf;
    for i = validFeatureNo
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