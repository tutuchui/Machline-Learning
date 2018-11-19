function [best_feature,best_threshold] = chooseAttribute(features,targets,validFeatureNo)
    examples = [features targets];
    maxGainInfo = -inf;
    for i = validFeatureNo
       % Make all the feature value of selected feature to be the candidate
       % thresholds, make sure all the splitting situations are tested for
       % choose the best threshold.
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
end