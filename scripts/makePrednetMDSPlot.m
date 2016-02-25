function makePrednetMDSPlot

%f_name=['../final_results/MDS_data_ndim2_v2.mat'];
f_name=['../final_results/MDS_data.mat'];
data=load(f_name);
vals.speed=data.speed;
vals.angle=data.angle;
vals.pca_1=data.pca_1;
X=data.X;
idx=[1:500,length(X)-500+1:length(X)];
X=data.X(idx,:);

x_min=min(X(:,1));
x_max=max(X(:,1));
y_min=min(X(:,2));
y_max=max(X(:,2));
s=std(X(:,1));

%epochs=[0,5,25,50,150];
epochs=[0,150];
n_points=500;

figure('Position',[786         324        1178         944]);
colormap('parula')
    count=0;
for v={'speed','pca_1'}
    var=v{1};

    these_vals=vals.(var);
    [~,~,rank]=unique(these_vals);
    for i=1:length(epochs)
        subplot(2,length(epochs),i+count);
        idx=(i-1)*n_points+1:i*n_points;
        scatter(X(idx,1),X(idx,2),100,rank,'filled');
        xlim([x_min-0.25*s,x_max+0.25*s])
        ylim([y_min-0.25*s,y_max+0.25*s])
        axis square
        axis off
        title(['Epoch ' num2str(epochs(i))])
    end
    count=count+length(epochs);
    %saveas(gcf,['../final_results/MDS_plot_' var '.tif']);
end









end