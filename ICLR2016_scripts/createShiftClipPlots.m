function createShiftClipPlots

load('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/val_sequences.mat');

to_plot=[27,1,6];

figure;
for i=to_plot
    subplot(3,1,1)
    imshow(squeeze(frames(i,1,1,:,:)));
    subplot(3,1,2)
    imshow(squeeze(frames(i,2,1,:,:)));
    subplot(3,1,3)
    imshow(squeeze(frames(i,3,1,:,:)));
    saveas(gcf,['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/sequence_' num2str(i-1) '.jpg']);
end

figure
for r=[6,17]
    load(['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/run' num2str(r) '_predictions.mat']);
    for i=to_plot
        imshow(squeeze(predictions(i,2,1,:,:)))
        saveas(gcf,['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/run' num2str(r) '_im' num2str(i-1) '.jpg']);
    end
end




end