function testVGGmodel

% need to run this beforehand
%run /home/bill/Libaries/matconvnet/matlab/vl_setupnn

net = load('/home/bill/Data/matconvnet_models/imagenet-vgg-verydeep-16.mat') ;

% obtain and preprocess an image
im = imread('/home/bill/Libraries/caffe/examples/images/cat.jpg') ;
im_ = single(im) ; % note: 255 range
ny = size(im_,1);
nx = size(im_,2);
if nx>ny
    d=round((nx-ny)/2);
    im_ = im_(:,d:d+ny-1,:);
elseif ny>nx
    d=round((ny-nx)/2);
    im_ = im_(d:d+nx-1,:,:);
end
% am just cropping for hand testing but would really do imresize
n=floor((min(nx,ny)-224)/2);
im_=im_(n:n+223,n:n+223,:);
%im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
%im_ = im_ - net.normalization.averageImage ;

tmp=load('test_rand_im.mat');
im_=tmp.im;

%patch1=zeros(
%im_(1:2,1:2,:)
%sum(sum(sum(im_(1:3,1:3,:).*net.layers{1}.weights{1}(:,:,:,1))))+net.layers{1}.weights{2}(1);

% run the CNN
res = vl_simplenn(net, im_) ;
disp(res(2).x(2:5,2:5,1))


% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.classes.description{best}, best, bestScore)) ;

w = squeeze(net.layers{1,1}.weights{1}(:,:,:,1));
b = net.layers{1,1}.weights{2}(1);
im_small=im_(1:3,1:3,:);

val=res(2).x(2,2,1);
disp(val)
disp(sum(sum(sum(w.*im_small)))+b)
end