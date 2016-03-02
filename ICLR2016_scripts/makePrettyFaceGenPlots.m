function makePrettyFaceGenPlots(run_num)

base_dir=['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/facegen_GAN_runs/run_' num2str(run_num) '/'];
f_name=[base_dir 'predictions.mat'];
load(f_name);

out_dir=[base_dir 'matplots/'];
mkdir(out_dir);

n_plot=20;

figure;
for i=1:n_plot
    subplot(1,4,1)
    imshow(squeeze(pre_sequences(i,1,1,:,:)))
    xlabel('T-2')
    
    subplot(1,4,2)
    imshow(squeeze(pre_sequences(i,2,1,:,:)))
    xlabel('T-1')
    
    subplot(1,4,3)
    imshow(squeeze(actual_sequences(i,1,1,:,:)))
    xlabel('Actual')
    
    subplot(1,4,4)
    imshow(squeeze(predictions(i,1,1,:,:)))
    xlabel('Prediction')
    
    title(['Run ' num2str(run_num) ' valclip ' num2str(i)])
    
    saveas(gcf,[out_dir 'valclip_' num2str(i) '.jpg']) 
end



end