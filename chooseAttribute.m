function [best_feature,best_threshold] = chooseAttribute(features,targets,validFeatureNo)
    examples = [features targets];
    maxGainInfo = -inf;
    for i = validFeatureNo
       thresholds  = features(:,i);
       for j = 1 : size(thresholds,1)
           threshold = round(thresholds(j,1)) - 1;
           gainInfo = calculateGainInfo(i,examples,threshold);
           if gainInfo > maxGainInfo
               candidate_feature = i;
               candidate_threshold = threshold;
               maxGainInfo = gainInfo;
           end
       end
    end
    best_feature = candidate_feature;
    best_threshold = candidate_threshold;
    disp(['f',num2str(best_feature)]);
    disp(gainInfo);
    
end