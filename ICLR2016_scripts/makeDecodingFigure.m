function makeDecodingFigure

[~,~,data]=xlsread('../final_results/Decoding_by_epoch_run67.xlsx');
d.epochs=cell2mat(data(2:end,2));
d.vars=data(2:end,3);
d.output_nums=cell2mat(data(2:end,4));
d.timesteps=cell2mat(data(2:end,5));
d.scores=cell2mat(data(2:end,6));


% vars={'pan_angles','Angle';
%     'pan_angular_speeds','Speed';
%     'pca_1','PCA 1';
%     'pca_2','PCA 2';
%     'pca_3','PCA 3';
%     'pca_4','PCA 4'};

vars={'pan_initial_angles','Initial Angle';
    'pca_1','PC 1';
    'pca_2','PC 2';
    'pan_angular_speeds','Speed';
    'pca_3','PC 3';
    'pca_4','PC 4'};

ylims={[0.98,1],[0.8, 0.89],[0.7,0.85],[0.9, 1],[0.64, 0.74],[0.45,0.75]};
yticks={[0.98,0.99,1],[0.8,0.83,0.86,0.89],[0.7,0.75,0.8,0.85],[0.9:0.02:1],[0.64:0.02:.74],[0.45:0.05:0.75]};
yticklabels={{'0.98','0.99','1.0'},{},{},{'0.90','0.92','0.94','0.96','0.98','1.0'},{},{}};

output_nums=[0,1];
timestep=4;
epochs=0:150;

scores={};
for vi=1:size(vars,1);
    v=vars{vi,1};
    these_scores=[];
    for o=output_nums
        vals=zeros(1,length(epochs));
        for e=1:length(epochs)
            idx=find(d.epochs==epochs(e) & strcmp(d.vars,v) & d.output_nums==o & d.timesteps==timestep);
            vals(e)=d.scores(idx);
        end
        these_scores=[these_scores;vals];
    end
    scores{end+1,1}=these_scores;
end

figure('Position',[147         111        1196         756]);
%h=tight_subplot(2,3,0.03,0.04,0.04);
for i=1:size(vars,1)
    %axes(h(i));
    subplot(2,3,i);
    hold on
    plot(epochs,scores{i}(1,:),'Color','b','LineWidth',2.5);
    plot(epochs,scores{i}(2,:),'Color','r','LineWidth',2.5);
    hold off
    if i==4
        legend({'hidden state','cell state'},'Location','SouthEast','FontSize',13)
        legend boxoff
        xlabel('Epoch','FontSize',15)
        ylabel('Decoding Accuracy (r^2)','FontSize',15)
    end
    set(gca,'TickDir','out');
    set(gca,'TickLength',[0.01 0.01]);
    set(gca,'LineWidth',2,'FontWeight','Bold','FontSize',13)
    title(vars{i,2},'FontSize',19)
    ylim(ylims{i})
    set(gca,'YTick',yticks{i});
    if ~isempty(yticklabels{i})
        set(gca,'YTickLabel',yticklabels{i})
    end
end
set(gcf,'Color','w');
export_fig(['../final_results/actual_figures/decoding_plot.pdf']);
%print(['../final_results/actual_figures/decoding_plot2'],'-dpdf');






end