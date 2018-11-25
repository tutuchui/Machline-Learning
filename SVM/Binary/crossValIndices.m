function indicies = crossValIndices(targets,fold)
indicies = zeros(length(targets), 1);
index = 1;
for i = 1:length(targets)
    indicies(i, 1) = index;
    index = index + 1;
    if (index == fold + 1)
        index = 1;
    end
end
indicies = indicies(randperm(size(indicies,1)),:);
end