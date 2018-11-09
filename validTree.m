function class = validTree(tree,sample)
%     x = test_inputs(1,:);
        while strcmp(tree.class,'null')
            attr_val = sample(tree.attribute);
            if attr_val >= tree.threshold
                tree = tree.kids{1,1};
            else
                tree = tree.kids{1,2};
            end
        end
        class = tree.class;
end