function prunedTree = pruneNode(tree,targetPrunedNode) 
    if strcmp(tree.op,targetPrunedNode)
        tree.kids = [];
        tree.class = tree.majorityValue;
        tree.threshold = 'null';
        tree.attribute = 'null';
    elseif ~isempty(tree.kids)
        tree.kids{1,1} = pruneNode(tree.kids{1,1},targetPrunedNode);
        tree.kids{1,2} = pruneNode(tree.kids{1,2},targetPrunedNode);
    end
    
    prunedTree = tree;
end