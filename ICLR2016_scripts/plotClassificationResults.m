function plotClassificationResults

plot_GAN=1;
version=5;

f_name=[getDropboxDir 'Cox_Lab/Predictive_Networks/results/Classification_Summary_' num2str(version) '.xlsx'];
data=xlsread(f_name);
if plot_GAN
    u_epochs=unique(data(:,2));
    u_epochs=u_epochs';
else
    u_epochs=setdiff(unique(data(:,2)),-1);
end
u_train=unique(data(:,4));
l_strs={};
for e=u_epochs
    l_strs{end+1,1}=num2str(e);
end
if plot_GAN
    l_strs{1}='GAN';
end
colors=zeros(length(u_epochs),3);
if plot_GAN
    colors(1,:)=[0,1,0];
    colors(2,:)=[1,0,0];
else
    colors(1,:)=[1,0,0];
end
n=length(u_epochs);
c0=0.3;
if plot_GAN
    offset=3;
else
    offset=2;
end
for i=offset:length(u_epochs)
    c=((1-c0)/(n-offset))*(i-offset)+c0;
    if i==n
        colors(i,:)=[0,0,1];
    else
        colors(i,:)=1-c*[1,1,1];
    end
end

for t=[0,4,-1]
    mean_scores=zeros(length(u_epochs),length(u_train));
    for i=1:length(u_epochs)
        for j=1:length(u_train)
            if t==-1
                if u_epochs(i)==0
                    tt=0;
                else
                    tt=4;
                end
            else
                tt=t;
            end
            idx=find(data(:,2)==u_epochs(i) & data(:,4)==u_train(j) & data(:,3)==tt);
            mean_scores(i,j)=data(idx,5);
        end
    end
    clf
    hold on
    for i=1:length(u_epochs)
        plot(mean_scores(i,:),'MarkerSize',18','Marker','.','LineWidth',3,'Color',colors(i,:),'MarkerEdgeColor',colors(i,:),'MarkerFaceColor',colors(i,:));
    end
    xlabel('# Training Examples')
%     if strcmp(var_str,'_angles')
%         ylim([90 100])
%     else
        ylim([0 100])
    %end
    ylabel('Classification Score (%)')
    if t==-1
        title({'Decoding Performance by Epoch'});
    else
        title({'Decoding Performance by Epoch';['timestep=' num2str(t)]});
    end
    legend(l_strs,'Location','best')
    if plot_GAN
        s='_withGAN';
    else
        s='';
    end
    if t==-1
        saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_epoch_total' s '_' num2str(version) '.tif']);
    else
        saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_epoch_t' num2str(t) s '_' num2str(version) '.tif']);
    end
    hold off
end











end