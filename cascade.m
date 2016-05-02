function hhh = cascade(features, weight, label, Npos, ...
    errorMax, downError,fprThreshold)
error = 1;
Nimgs=size(weight,1);
examples = features;
t=0;

while (error > errorMax)
    %number of boostings
    t=t+1;
    
    hh = boost(examples, weight, label, Npos, Nimgs, error,...
        min(downError,(error-errorMax)),...
        fprThreshold);
    %only pick the ones that are predicted 1
    idx=1:size(examples,2);
    posIdx = idx(hh.bestResult==1);
    
    t
    %update the examples, weights and label
    examples = examples(:,posIdx);
    weight = 1.0/length(posIdx)*ones(length(posIdx),1);
    label = label(posIdx,:);
    error = hh.error;
    
    
    if t == 1
        hhh = hh;
    else 
        hhh(t) = hh;
    end
end
    
    
 