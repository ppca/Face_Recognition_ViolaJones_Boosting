% Harr-like feature extraction for one image
% 
% Input
% filepath: string, the name of the directory containing the faces/ and background/
% directories
% row, col: integers, dimensions of the training images 
% Npos: int, number of face images 
% Nneg: int, number of background images
% 
% Output
% features: ndarray, extracted Haar features
function [ features, Npos, Nneg ] = getHaar(filePath)
row = 64; col = 64;  % 64x64 pixel images
posFilePath = [filePath 'faces/' ];
negFilePath = [filePath 'background/'];
posImg = loadImagesAdaBoost(posFilePath, row, col);
negImg = loadImagesAdaBoost(negFilePath, row, col);

% get total number of images
Nimg = size(posImg,3) + size(negImg,3);
Npos = size(posImg,3);
Nneg = size(negImg,3);
Nfeatures = 295937;
features = zeros(Nfeatures, Nimg);
for i = 1:Nimg
    if i <= size(posImg,3)
        % convert to integral image
        intImg = zeros(row+1,col+1);
        intImg(2:row+1,2:col+1) = cumsum(cumsum(posImg(:,:,i)),2);
        % compute features
        features(:,i) = computeFeature(intImg,row,col);
    else
        % convert to integral image
        intImg = zeros(row+1,col+1);
        intImg(2:row+1,2:col+1) = cumsum(cumsum(negImg(:,:,i-Npos)),2);
        % compute features
        features(:,i) = computeFeature(intImg,row, col);
    end
end
features (1001,:)
%optional
features_adaboost.features = features;
features_adaboost.Npos = Npos;
features_adaboost.Nneg = Nneg;
%save ('features_adaboost_test.mat','features_adaboost','-mat','-v7.3');
save('features_adaboost_train.mat', 'features_adaboost', '-mat','-v7.3');

end
%% load images
function imgs = loadImagesAdaBoost(filePath, row, col)
% get images in 'filePath'
files = dir([filePath '*.jpg']);
imgs = zeros(row,col,length(files));

for i = 1: length(files)
    files(i)
    img = imread([filePath files(i).name]);
    imgGray = double(rgb2gray(img));
    imgs(:,:,i) = imgGray;
end
end



%% compute Haar features
function feature = computeFeature(I, row, col)
feature = zeros(295937,1);

%extract horizontal feature
cnt = 1;
window_h = 1; window_w=2; %window size 
for h = 1:row/window_h %extend the size of one rectangular
    for w = 1:col/window_w
        for i = 1:4:row+1-h*window_h %slide 
            for j = 1:4:col+1-w*window_w
                rect1=[i,j,w,h];
                rect2=[i,j+w,w,h];
                %if cnt>10
                %    return
                %end
                feature(cnt)=sumRect(I, rect2)- sumRect(I, rect1);
                
                cnt=cnt+1;
            end
        end
    end
end


window_h = 2; window_w=1; %window size 
for h = 1:row/window_h
    for w = 1:col/window_w
        for i = 1:4:row+1-h*window_h
            for j = 1:4:col+1-w*window_w
                rect1=[i,j,w,h];
                rect2=[i+h,j,w,h];
                feature(cnt)=sumRect(I, rect1)- sumRect(I, rect2);
                cnt=cnt+1;
            end
        end
    end
end
end

%%
function [rectsum] = sumRect(I, rect_four) 
% given four corner points in the integral image 
% calculate the sum of pixels inside the rectangular. 
row_start = rect_four(1); 
col_start = rect_four(2); 
width = rect_four(3);
height = rect_four(4); 
one = I(row_start, col_start); 
two = I(row_start, col_start+width); 
three = I(row_start+height, col_start); 
four = I(row_start+height, col_start+width); 
rectsum = four + one -(two + three);
end