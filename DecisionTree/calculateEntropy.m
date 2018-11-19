function entropy = calculateEntropy(labels)
if size(labels,1) == 0
    entropy = 0;
elseif all (~(diff(labels)))
        entropy = 0;
else
    n = sum(xor(labels(:,1),1));
    p = sum(xor(labels(:,1),0));
    Pp = p/(p+n);
    Pn = n/(p+n);
    entropy = -Pp * log2(Pp) - Pn * log2(Pn);
end