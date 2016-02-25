function runThruPlots


load(['../final_results/facegen_predictions_submission2_635best.mat']);

plot_idx=1:200;
%p_idx=[4,38,70,85,103,104,111,114,118,134,136,140,142,149,156,178,179,180,184];
%plot_idx=[38,70,103,114,136,149,184];
%plot_idx=setdiff(p_idx,plot_idx);
%plot_idx=[111,118,134,140,179,180];
%plot_idx=[21,43,41,49,52,126,56,69,87,88,175,178,168];
plot_idx=[23,27,40,53,68,76,78,85];
X=squeeze(X(plot_idx,:,:,:,:));
predictions_GAN=squeeze(predictions_GAN(plot_idx,:,:,:,:));
predictions_MSE=squeeze(predictions_MSE(plot_idx,:,:,:,:));

%figure('Position',[236,904,1798,389])
for i=1:length(plot_idx)
    figure('Position',[236,904,1798,389])
    for j=4:6 
        subplot(1,5,j-3)
        imshow(squeeze(X(i,j,:,:)));
    end
    title(plot_idx(i))
    subplot(1,5,4)
    imshow(squeeze(predictions_MSE(i,:,:)));
    subplot(1,5,5)
    imshow(squeeze(predictions_GAN(i,:,:)));
    pause
end






end


