function makePlotsForIan

load(['../final_results/facegen_predictions_submission2_635best.mat'])

%plot_idx=[70,111,118];
plot_idx=[57,68,56];

X=squeeze(X(plot_idx,:,:,:,:));
predictions_GAN=squeeze(predictions_GAN(plot_idx,:,:,:,:));
predictions_MSE=squeeze(predictions_MSE(plot_idx,:,:,:,:));
val.dx1=100;
val.dx2=100;
for i=1:length(plot_idx)
    for j=1:6
        for k=1:2
            v=min(squeeze(X(i,j,:,:)),[],k);
            rval=find(v<1,1,'first');
            lval=size(X,3)-find(v<1,1,'last');
            val.(['dx' num2str(k)])=min(val.(['dx' num2str(k)]),rval);
            val.(['dx' num2str(k)])=min(val.(['dx' num2str(k)]),lval);
        end
    end
end
dx=val.dx1;
X=X(:,:,:,dx-1:151-dx);
predictions_MSE=predictions_MSE(:,:,dx-1:151-dx);
predictions_GAN=predictions_GAN(:,:,dx-1:151-dx);

figure('Position',[6     2   962   834])
h=tight_subplot(3,4,[0.00 0.00],[0.00 0.00],[0.00 0.00]);
hold on
for i=1:3
    axes(h((i-1)*4+1));
    imshow(squeeze(X(i,6,:,:)));
    axes(h((i-1)*4+2));
    imshow(squeeze(predictions_MSE(i,:,:)));
    axes(h((i-1)*4+3));
    imshow(squeeze(predictions_GAN(i,:,:)));
    axes(h((i-1)*4+4));
    imshow(ones(150));
end
hold off
set(gcf,'color','w')

export_fig(['../plots_for_Ian_submission2/PGN_grid.pdf'],gcf)
export_fig(['../plots_for_Ian_submission2/PGN_grid.tif'],gcf)

figure('Position',[3           2        1127         834])
h=tight_subplot(1,1,[0.00 0.00],[0.00 0.00],[0.00 0.00]);
%figure('units','normalized','outerposition',[0 0 1 1])
for i=1:3
    axes(h(1))
    imshow(squeeze(X(i,6,:,:)));
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_GT.pdf'],gcf)
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_GT.tif'],gcf)
    axes(h(1))
    imshow(squeeze(predictions_MSE(i,:,:)));
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_MSE.pdf'],gcf)
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_MSE.tif'],gcf)
    axes(h(1))
    imshow(squeeze(predictions_GAN(i,:,:)));
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_AL.pdf'],gcf)
    export_fig(['../plots_for_Ian_submission2/PGN_face' num2str(i) '_AL.tif'],gcf)
end

end