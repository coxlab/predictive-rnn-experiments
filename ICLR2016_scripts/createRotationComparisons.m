function createRotationComparisons


load('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/MNIST_rotation_run27_predictions.mat');

outdir=['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/MNIST_rotation_run27_predictions/'];
mkdir(outdir);

to_plot=[32,37,28,39];

figure
for i=to_plot
    subplot(2,1,1);
    imshow(squeeze(actual(i,1,1,:,:)));
    %title('Actual')
    subplot(2,1,2);
    imshow(squeeze(predictions(i,1,1,:,:)));
    %title('Predicted')
    saveas(gcf,[outdir 'val_im' num2str(i) '_t10.jpg']);
end



to_vid=37;

for i=to_vid
    
    writer=VideoWriter([outdir 'val_clip' num2str(i) '.avi']);
    writer.FrameRate=10;
    open(writer);
    for j=1:5
        writeVideo(writer,squeeze(predictions(i,j,1,:,:)));
    end
    close(writer)
end







end