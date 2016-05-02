% sweep throught image by stride w_h, w_v to get sub-windows
% apply cascade classifiers on all sub-windows for face detection
function out = violaTest(img, w_h, w_v, hhh)

[Nrow, Ncol] = size(img);
row = 64;
col = 64;
N=size(hhh,2);
features=[];
out=[];
for i = 1:w_v:(Nrow-row+1)
    for j = 1:w_h:(Ncol-col+1)
        intImg = zeros(row+1,col+1);
        intImg(2:row+1,2:col+1) = cumsum(cumsum...
            (img(i:(i+row-1),j:(j+col-1))),2);
        % compute features
        features(:,j)= computeFeature(intImg,row,col);
    end
    Npics=size(features,2);
    
    % apply cascade classifier
    fff=ones(1,Npics);
    for k=1:N
        hh=hhh(k);
        fff = fff.*((hh.alpha*(repmat(hh.p',1,Npics).*(features(hh.featureIdx,:)-...
            repmat(hh.theta',1,Npics))>0)-hh.theta2)>0);
       
        if (all(fff==0))
            break
        end
    end
    idx = find(fff>0);
    out = [out; [idx',repmat(i,length(idx),1)]];
    
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