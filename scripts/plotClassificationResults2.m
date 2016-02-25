function plotClassificationResults2

version=2;

to_plot={65,300,4,'Predictive RNN';
    98,300,4,'Autoencoder RNN';
    110,300,0,'Autoencoder FC';
    134,300,0,'Autoencoder FC 4096';
    120,125,4,'Autoencoder RNN all images';
    133,113,0,'Autoencoder FC all images';
    135,86,0,'Autoencoder FC 4096 all images';
    67,0,0,'Initial Weights';
    307,-1,4,'Predictive GAN'};

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
plot(scores','MarkerSize',18,'Marker','.','LineWidth',3);
legend(to_plot(:,4),'Location','SouthEast')
ylim([0 100])
xlabel('# Training Examples')
ylabel('Classification Score (%)')
title('Classification Performance')

%saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_model_' num2str(version) '.tif']);
print([getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_model_' num2str(version)],'-dpdf');





end