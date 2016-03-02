function plotPCADecoding(run_pca)

[~,~,data]=xlsread('../final_results/Decoding_by_epoch_run67.xlsx');
d.epochs=cell2mat(data(2:end,2));
d.vars=data(2:end,3);
d.output_nums=cell2mat(data(2:end,4));
d.timesteps=cell2mat(data(2:end,5));
d.scores=cell2mat(data(2:end,6));

if run_pca
vars={'pca_1','PC 1';
    'pca_2','PC 2'};
%     'pca_3','PC 3'};
%     'pca_4','PC 4';
%     'pca_5','PC 5'};
epochs=0:150;
else
vars={'pan_initial_angles','Initial Angle';
    'pan_angular_speeds','Speed'};
epochs=0:50;
end

output_num=0;
timestep=4;


figure;
scores=zeros(size(vars,1),length(epochs));
for vi=1:size(vars,1);
    for ei=1:length(epochs)
        idx=strcmp(d.vars,vars{vi,1}) & d.output_nums==output_num & d.timesteps==timestep & d.epochs==epochs(ei);
        scores(vi,ei)=d.scores(idx);
    end
end

plot(epochs,scores,'LineWidth',2);
ylabel('Decoding Accuracy (r^2)','FontSize',12)
xlabel('Training Epoch','FontSize',12)
legend(vars(:,2),'Location','SouthEast','FontSize',12)
legend boxoff
set(gca,'LineWidth',1.5,'FontWeight','Bold')
set(gca,'TickDir','out');
set(gca,'TickLength',[0.01 0.01]);
set(gcf,'color','w')
box(gca,'off');
if ~run_pca
    ylim([0.9 1]);
    set(gca,'YTick',[0.9:0.02:1])
else
    ylim([0.7 0.9]);
    set(gca,'YTick',[0.7:0.05:0.9])
end
if run_pca
export_fig(['../final_results/cosyne_figure/pca_decoding.tif'])
else
  export_fig(['../final_results/cosyne_figure/angle_decoding.tif'])  
end

end