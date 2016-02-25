function makeFig2

load(['../final_results/facegen_predictions_submission2_635best.mat'])

%plot_idx=[4,142,70,149,118,111];
%plot_idx=[70,4,111,149,118,142];
%plot_idx=[134,15,129,76,178,94]; %for 668 
plot_idx=[57,9,68,31,56,85];
pop_idx=[1,3,5];
%pop_idx=[2,3,4];
n_pre=3;
n_plot=length(plot_idx);

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
figure('Position',[100 100 324*2 329*2]);
%figure('Position',[95 95 334 344]);
h=tight_subplot(n_plot,n_pre+4,[0.0 0.0],[0.00 0.00],[0.00 0.00]);
hold on
for i=1:n_plot
    for j=6-n_pre:6
        axes(h((n_pre+4)*(i-1)+j-n_pre+1));
        imshow(squeeze(X(i,j,:,:)));
    end
    axes(h((n_pre+4)*(i-1)+n_pre+2));
    imshow(squeeze(predictions_MSE(i,:,:)));
    axes(h((n_pre+4)*(i-1)+n_pre+3));
    imshow(squeeze(predictions_GAN(i,:,:)));
    axes(h((n_pre+4)*(i-1)+n_pre+4));
    imshow(ones(150));
end
hold off
set(gcf,'color','w')

export_fig(['../final_results/actual_figures/fig2_grid_submission2.eps'],gcf)
export_fig(['../final_results/actual_figures/fig2_grid_submission2.pdf'],gcf)


%figure('Position',[100 100 429 300]);
figure('Position',[100   100   595   510]);
h=tight_subplot(3,4,[0.00 0.00],[0.00 0.00],[0.00 0.00]);
for i=1:length(pop_idx)
    axes(h(1+(i-1)*4));
    imshow(squeeze(X(pop_idx(i),6,:,:)));
    axes(h(2+(i-1)*4));
    imshow(squeeze(predictions_MSE(pop_idx(i),:,:)));
    axes(h(3+(i-1)*4));
    imshow(squeeze(predictions_GAN(pop_idx(i),:,:)));
     axes(h(4+(i-1)*4));
    imshow(ones(150));
    
    
end
set(gcf,'color','w')
export_fig(['../final_results/actual_figures/fig2_popout_submission2.eps'],gcf)
export_fig(['../final_results/actual_figures/fig2_popout_submission2.pdf'],gcf)


end