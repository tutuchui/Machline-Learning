function gainInfo = calculateGainInfo(featureNo,examples,threshold)
%Function calculateGainInfo return the gainInfo by using the selected
%featureNo and threshold
    e1 = examples(:,featureNo) >= threshold;
    e2 = ~e1;
    E1 = examples(e1,:);
    E2 = examples(e2,:);
    labelIdx = size(examples,2);
    
    p1 = sum(xor(E1(:,labelIdx),0));
    n1 = sum(xor(E1(:,labelIdx),1));
    p2 = sum(xor(E2(:,labelIdx),0));
    n2 = sum(xor(E2(:,labelIdx),1));
    N = size(examples,1);
    entropyE1 = calculateEntropy(E1(:,labelIdx));
    entropyE2 = calculateEntropy(E2(:,labelIdx));
    remainder = (p1+n1)/N * entropyE1 + (p2+n2)/N * entropyE2;
    gainInfo = calculateEntropy(examples(:,labelIdx)) - remainder;
end