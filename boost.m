function hh = boost(examples, weight, label, Npos, Nimgs, lastError,...
    downError,fprThreshold)
t=1;
theta=[];
p=[];
Idx=[];
error = 1;
alpha=[];
pred=[];
currentMin=[];
fpr=1;
%get weak classifier and reweight examples until a certain decrease in FPR
while ((fpr>fprThreshold) || ((lastError-error)<downError))
    
    h = getWeakClassifier(examples,weight,label,Npos);
    theta(t)=h.theta;
    p(t)=h.p;
    Idx(t)=h.featureIdx;
    pred(:,t)=h.bestResult;
    err = h.currentMin;
    currentMin(t) = err;
    
    %calculate beta and reweight examples
    beta = err/(1-err);
    weight = weight.*beta.^(pred(:,t)==label);
    weight = weight/sum(weight);
    
    %calculate alpha
    alpha(t) = -log(beta);
    
    %calculate meta classifier up till now and its threshold
    f = pred*alpha';
    m = min(f(label==1,:));
    if m == min(f)
        theta2 = m-0.01;
    else
        theta2 =(max(f(f<m,:))+m)/2;
    end
    
    %calculate false positive rate
    fpr = mean(label(f>theta2,:)==0);
    error = fpr*sum(f>theta2)/Nimgs;
    
    t=t+1;
end

hh.theta = theta;
hh.currentMin = currentMin;
hh.p = p;
hh.featureIdx = Idx;
hh.fpr = fpr;
hh.alpha = alpha;
hh.theta2 = theta2;
hh.bestResult = f>theta2;
hh.error=error;
end
    

