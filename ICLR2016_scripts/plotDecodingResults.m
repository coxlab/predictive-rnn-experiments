function plotDecodingResults

[~,~,data]=xlsread('../final_results/Decoding_by_epoch_run67.xlsx');
d.epochs=cell2mat(data(2:end,2));
d.vars=data(2:end,3);
d.output_nums=cell2mat(data(2:end,4));
d.timesteps=cell2mat(data(2:end,5));
d.scores=cell2mat(data(2:end,6));


vars={'pan_angles','Angle';
    'pan_angular_speeds','Speed';
    'pca_1','PCA 1';
    'pca_2','PCA 2';
    'pca_3','PCA 3';
    'pca_4','PCA 4';
    'pca_5','PCA 5'};

output_nums=[0,1];
timesteps=[0,4];
epochs=0:150;

figure;
for vi=1:size(vars,1);
    v=vars{vi,1};
    l_strs={};
    scores=[];
    outs=[];
    ts=[];
    for o=output_nums
        for t=timesteps
            if strcmp(v,'pan_angular_speeds')&&t==0
                continue
            end
            vals=zeros(1,length(epochs));
            if o==0
                o_str='h';
            else
                o_str='c';
            end
            l_strs{end+1,1}=[o_str ' t=' num2str(t)];
            outs(end+1,1)=o;
            ts(end+1,1)=t;
            for e=1:length(epochs)
                idx=find(d.epochs==epochs(e) & strcmp(d.vars,v) & d.output_nums==o & d.timesteps==t);
                vals(e)=d.scores(idx);
            end
            scores=[scores;vals];
        end
    end
    
    clf
    hold on
    for s=1:size(scores,1)
        if outs(s)==0
%             if ts(s)==0
%                 base_c=[0.5 0.5 1];
%             elseif ts(s)==2
%                 base_c=[0.25 0.25 1];
%             elseif ts(s)==4
%                 base_c=[0 0 1];
%             end
            base_c=[0 0 1];
        else
%             if ts(s)==0
%                 base_c=[1 0.5 0.5];
%             elseif ts(s)==2
%                 base_c=[1 0.25 0.25];
%             elseif ts(s)==4
%                 base_c=[1 0 0];
%             end
            base_c=[1 0 0];
        end
        if ts(s)==0
            l='-.';
        elseif ts(s)==2
            l='-.';
        else
           l='-'; 
        end
       
        plot(epochs,scores(s,:),'Color',base_c,'LineWidth',2,'LineStyle',l);
    end
    set(gca,'TickDir','out');
    set(gca,'TickLength',[0.01 0.01]);
    ylabel('Decoding Accuracy (r^2)')
    xlabel('Training Epoch')
    title(['Decoding ' vars{vi,2} ])
    legend(l_strs,'Location','SouthEast')
    set(gca,'LineWidth',2,'FontWeight','Bold')
    hold off
    print(['../final_results/Decoding_by_epoch_' v],'-dpdf')
    print(['../final_results/Decoding_by_epoch_' v],'-dpng')
end


close('all')






end