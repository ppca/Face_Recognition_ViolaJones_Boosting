%% Compute features for all 4000 images
pic_path='C:\Users\Administrator\Google ÔÆ¶ËÓ²ÅÌ\fall 2015\ML\project1\train_test\';
tic
[ features, Npos, Nneg ] = getHaar(pic_path);
toc

%% train model on 2000 images
tic
train = [features(:,1:1000) features(:,2001:3000)];
test = [features(:,1001:2000) features(:,3001:4000)];
label=[ones(1000,1);zeros(1000,1)];
hhh_2000_train = cascade(trian, ones(2000,1)*1.0/2000, label, ...
    1000, 0.001, 0.05,0.3);
save hhh_2000_allfeatures.mat hhh_2000_train;
toc

%% test cascade on test data
fff=ones(1,2000);
fpr_list=[];
dr=[];
fneg_list=[];
label=[ones(1000,1);zeros(1000,1)];
for k=1:size(hhh_2000_train,2)
    hh=hhh_2000_train(k);
    % zero value in any layer of cascade lead to rejection
    % so element-wise multiply each layer's result
    fff = fff.*((hh.alpha*(repmat(hh.p',1,2000).*...
        (test(hh.featureIdx,:)-...
        repmat(hh.theta',1,2000))>0)-hh.theta2)>0);
    %false positives
    fpr_list(k)=mean(label(fff>0,:)==0);
    %detection rate
    dr(k)= mean(fff'==label);
    fneg_list(k)=mean(label(fff<=0,:)==1);
end

%table the test errors of each stage
testResult.fpr=fpr_list';
testResult.DR=dr';
testResult=struct2table(testResult);
error=1-mean(fff'==label);

%% face detection with cascade on class image

classImg = imread('class.jpg');
tic 
% out11 keeps all upper left coordinates of face windows
out11= violaTest(double(classImg),20,20,hhh_2000_train);
toc

I=imread('class.jpg');

RGB = repmat(I,[1,1,3]);

imshow(RGB);
usedIdx=[];

%% display face sub-windows on class image
for i=1:size(out11,1)
    if any((usedIdx-i)==0)==0
        pos=out11(i,:);
        idxLeft=1:size(out11,1);
        idxLeft(usedIdx)=[];
        sss=abs(repmat(pos,length(idxLeft),1)-...
            out11(idxLeft,:));
        posIdx=idxLeft(all(sss<25,2));
        if length(posIdx)==1
            leftcorner=out11(posIdx,:);
            rightcorner=out11(posIdx,:)+64;
        else
            leftcorner=mean(out11(posIdx,:));
            rightcorner=mean(out11(posIdx,:)+64);
        end
        if length(posIdx)>3
            rectangle('Position',...
                [leftcorner, rightcorner-leftcorner], ...
                'LineWidth',2, 'EdgeColor','b');
            hold on
        end
        usedIdx=sort([usedIdx posIdx]);
    end
end
       
