function createRotationVids

load([getDropboxDir 'Cox_Lab/Predictive_Networks/misc/MNIST_rotation_val_clips.mat']);


nmake=10;

idx=randperm(50,nmake);
idx=union(idx,37);

for i=idx
    writer=VideoWriter([getDropboxDir 'Cox_Lab/Predictive_Networks/misc/example_rotation_clips/val_clip_' num2str(i-1) '.avi');
    open(writer);
    for j=1:10
        writeVideo(writer,clips(i,j,:,:));
    end
    close(writer)
end








end