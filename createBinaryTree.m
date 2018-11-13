function tree = createBinaryTree(op,kids,class,attribute,threshold,majorityValue)
    tree = struct('op',op,'kids',kids,'class',class,'attribute',attribute,'threshold',threshold,'majorityValue',majorityValue);
end