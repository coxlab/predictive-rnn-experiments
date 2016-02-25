

im=im2double(imread('/home/bill/Data/FaceGen_Rotations/clipset1/images_processed/face_0_frame_0.png'));
%im_filtered=butterworthbpf(im,0.1,200,4);
%imshow(im_filtered)
out_dir='/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/filtered_images_test/';
mkdir(out_dir);

figure
for n=[3,5,7]
    im_std=stdfilt(im,ones(n));
    for p=[1,0.9,0.75,0.5,1/3]
        this_im=im_std.^p;
        imshow(this_im);
        title(['n' num2str(n) ' p' num2str(p)])
        saveas(gcf,[out_dir 'imstdfilt_n' num2str(n) '_p' num2str(p) '.tif'])
    end
end
