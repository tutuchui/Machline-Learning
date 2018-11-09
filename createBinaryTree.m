function tree = createBinaryTree(op,kids,class,attribute,threshold)
    tree = struct('op',op,'kids',kids,'class',class,'attribute',attribute,'threshold',threshold);
end