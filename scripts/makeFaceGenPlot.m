function makeFaceGenPlot

n_plot=6;
n_pre=3;

load(['../final_results/facegen_predictions_submission2_663best.mat'])

%rng(0);

%perm=randperm(size(X,1));
%plot_idx=perm(1:n_plot);

%plot_idx=[174,34,152,101,103,171,183,182,134,38];
%plot_idx=[152,38,103,134];
%plot_idx=[152,103,171,134,38];
%plot_idx=[70,111,134,140,142,179,180];
%poss_idx=[4,38,70,85,103,104,111,114,118,134,136,140,142,149,156,178,179,180,184];
poss_idx=[4,38,85,103,104,134,136,140,149,156,178,179,180,184];
%n_plot=length(plot_idx);
perm=randperm(length(poss_idx));
plot_idx=zeros(n_plot,1);
plot_idx(3)=70;
plot_idx(5)=118;
plot_idx(2)=142;
plot_idx(6)=111;
plot_idx([1,4])=poss_idx(perm(1:2));

plot_idx=[4,142,70,149,118,111];
plot_idx=[70,4,111,149,118,142];

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
%figure('Position',[156 31 1143 1306]);
figure('Position',[151           1        1376        1333])
h=tight_subplot(n_plot,n_pre+4,[0.00 0.00],[0.00 0.00],[0.00 0.00]);
hold on
for i=1:n_plot
    for j=6-n_pre:6
        %subplot(n_plot,5,5*(i-1)+j-3)
        axes(h((n_pre+4)*(i-1)+j-n_pre+1));
        imshow(squeeze(X(i,j,:,:)));
    end
    axes(h((n_pre+4)*(i-1)+n_pre+2));
    %subplot(n_plot,5,5*(i-1)+4)
    imshow(squeeze(predictions_MSE(i,:,:)));
    axes(h((n_pre+4)*(i-1)+n_pre+3));
    %subplot(n_plot,5,5*(i-1)+5)
    imshow(squeeze(predictions_GAN(i,:,:)));
    axes(h((n_pre+4)*(i-1)+n_pre+4));
    imshow(ones(150));
end
hold off
set(gcf,'color','w')

%print(['../final_results/FaceGen_grid'],'-dps');
%print(['../final_results/FaceGen_grid'],'-deps');
%print(['../final_results/FaceGen_grid'],'-dpng');
%print(['../final_results/FaceGen_grid'],'-dpdf');
%print(['../final_results/FaceGen_grid'],'-ps');
%saveas(gcf,['../final_results/FaceGen_grid.ps']);

export_fig(['../final_results/actual_figures/FaceGen_grid_submission2.pdf'],gcf)
export_fig(['../final_results/actual_figures/FaceGen_grid_submission2.tif'],gcf)

%export_fig(['../final_results/FaceGen_grid.ps'],gcf)


end