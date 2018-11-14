function [Majority] = MAJORITY_VALUE(targets)
%MAJORITY_VALUE Returns the majority value in target array
MajNum = 0;
MajLabel = [];
for i = 1 : size(targets,1)
    CurFreq = sum(targets(:,1)==targets(i,1)); %Checks how often label at poistion occurs in whole
    
    if (CurFreq > MajNum) %updates majority label
        MajLabel = targets(i,1);
        MajNum = CurFreq;
    end
end

Majority = MajLabel; %returns the Majority Label
end

