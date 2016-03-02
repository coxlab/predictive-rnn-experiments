function makeProjectionsPlot


base_dir=['/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_67/'];

epochs=[0,5,25,50,150];
buf=0.03;
model_epoch_str='_modelepoch150';
model_epoch_str='';
%var_axis={'pan_angular_speeds','pca_1'};
var_axis={'pan_initial_angles','pca_1'};

for e=epochs
    this_dir=[base_dir 'feature_analysis_epoch' num2str(e) '/projections/'];
    data.(['epoch_' num2str(e)])=load([this_dir 'projections' model_epoch_str '.mat']);
    for v=var_axis
       x=data.(['epoch_' num2str(e)]).(v{1});
       data.(['epoch_' num2str(e)]).(v{1})=x-mean(x);
    end
end


total_min=inf;
total_max=-inf;
for v=var_axis
   min_val=inf;
   max_val=-inf;
   for e=epochs
      these_vals=data.(['epoch_' num2str(e)]).(v{1});
      min_val=min(min_val,min(these_vals));
      max_val=max(max_val,max(these_vals)); 
   end
   min_vals.(v{1})=min_val;
   max_vals.(v{1})=max_val;
   total_min=min(total_min,min_val);
   total_max=max(total_max,max_val);
end


plot_pairs={{'pan_initial_angles','pca_1'}};%,{'pan_initial_angles','pan_angular_speeds'}};

for i=1:length(plot_pairs)
    figure('Position',[116         468        2109         843]);
    %h=tight_subplot(2,length(epochs),[0.00 0.02],0.08,0.03);
    colormap('parula')
    h=tight_subplot(2,length(epochs),0.02,0.05,0.05);
    for j=1:2
        for ei=1:length(epochs)
            axes(h((j-1)*length(epochs)+ei));
            x=data.(['epoch_' num2str(epochs(ei))]).(plot_pairs{i}{1});
            y=data.(['epoch_' num2str(epochs(ei))]).(plot_pairs{i}{2});
            c=data.(['epoch_' num2str(epochs(ei))]).(plot_pairs{i}{j});
            [~,~,rank]=unique(c);
            hold on
            scatter(x,y,300,rank,'.');
            plot([0 0],[0 total_max/4],'k','LineWidth',2)
            plot([0 total_max/4],[0,0],'k','LineWidth',2)
            xlim([total_min-buf total_max+buf]);
            ylim([total_min-buf total_max+buf]);
            axis square
            
            hold off
            axis off
            if j==1
                title(['Epoch ' num2str(epochs(ei))])
            end
        end
        set(gcf,'color','w')
    end
    %export_fig(['../final_results/actual_figures/projection_plot_axis' model_epoch_str '.eps'],gcf)
    %export_fig(['../final_results/actual_figures/projection_plot_' plot_pairs{i}{1} '_' plot_pairs{i}{2}  model_epoch_str '.eps'],gcf)
    %print(['../final_results/actual_figures/projection_plot_' plot_pairs{i}{1} '_' plot_pairs{i}{2}],'-dpng')
end

% for v=var_axis
%     figure('Position',[173         865        2066         300]);
%     for ei=1:length(epochs)
%         x=data.(['epoch_' num2str(epochs(ei))]).(var_axis{1});
%         y=data.(['epoch_' num2str(epochs(ei))]).(var_axis{2});
%         z=data.(['epoch_' num2str(epochs(ei))]).(var_axis{3});
%         subplot(1,length(epochs),ei);
%         c=data.(['epoch_' num2str(epochs(ei))]).([v{1} '_truth']);
%         %scatter3(x,y,z,5,c,'.')
%         scatter(x,z,5,c)
%         xlim([total_min total_max]);
%         ylim([total_min total_max]);
%         if ei==1
%             title({['Epoch ' num2str(epochs(ei))], ['Colored by ' v{1}]})
%         else
%             title(['Epoch ' num2str(epochs(ei))])
%         end
%     end
% end



close('all')

end