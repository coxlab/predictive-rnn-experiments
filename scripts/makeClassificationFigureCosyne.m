function makeClassificationFigureCosyne

version=2;

to_plot={65,300,4,'Predictive RNN';
    120,-1,4,'Autoencoder RNN';
    133,-1,0,'Autoencoder FC (equal #units)';
    135,-1,0,'Autoencoder FC (equal #weights)';
    67,0,0,'Initial Weights'};

f_name=[getDropboxDir 'Cox_Lab/Predictive_Networks/results/Classification_Summary_' num2str(version) '.xlsx'];
data=xlsread(f_name);

n_train=1:10;

scores=zeros(size(to_plot,1),length(n_train));


for i=1:size(to_plot,1)
    for t=1:length(n_train)
        idx=find(data(:,1)==to_plot{i,1} & data(:,2)==to_plot{i,2} & data(:,3)==to_plot{i,3} & data(:,4)==n_train(t));
        scores(i,t)=data(idx,5);
    end
end

figure
plot(scores','MarkerSize',15,'Marker','.','LineWidth',2);
legend(to_plot(:,4),'Location','SouthEast','FontSize',11)
ylim([0 100])
legend boxoff
set(gca,'LineWidth',1.5,'FontWeight','Bold')
xlabel('# Training Angles','FontSize',12)
ylabel('Classification Score (%)','FontSize',12)
set(gca,'TickDir','out');
set(gca,'TickLength',[0.01 0.01]);
set(gcf,'Color','w');
box(gca,'off');
xlim([1 10])
export_fig([getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/cosyne_figure/Classification_by_model.tif'])




end