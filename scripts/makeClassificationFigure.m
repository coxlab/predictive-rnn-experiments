function makeClassificationFigure

version=2;

% to_plot={65,300,4,'Predictive RNN';
%     98,300,4,'Autoencoder RNN';
%     110,300,0,'Autoencoder FC';
%     134,300,0,'Autoencoder FC 4096';
%     120,-1,4,'Autoencoder RNN all images';
%     133,-1,0,'Autoencoder FC all images';
%     135,-1,0,'Autoencoder FC 4096 all images';
%     67,0,0,'Initial Weights';
%     307,-1,4,'Predictive GAN'};

to_plot={65,300,4,'PGN MSE';
    635,-1,4,'PGN AL/MSE';
    139,300,4,'AE LSTM (dynamic)';
    136,300,4,'AE LSTM (static)';
    137,300,0,'AE FC (= #units)';
    138,300,0,'AE FC (= #weights)'
    67,0,0,'LSTM Rand Weights'};
    
    
%     'pixels',-1,0,'pixels';
%     'lbp',-1,0,'LBP';
%     'hog',-1,0,'HOG'};
    %499,30,4,'PGN run 499';
    %511,70,4,'PGN run 511'};
    %478,300,4,'PGN MSE/AL';
    %135,-1,0,'AE FC (= #weights) stopped early';
    %143,100,0,'DAE10 FC (= #weights)';
    %142,107,0,'DAE20 FC (= #weights)'};
%     'pixels',-1,0,'pixels';
%     'lbp',-1,0,'LBP';
%     'hog',-1,0,'HOG';
%     67,0,0,'LSTM Rand Weights'};

f_name=[getDropboxDir 'Cox_Lab/Predictive_Networks/results/Classification_Summary_' num2str(version) '.xlsx'];
[data,str_data,~]=xlsread(f_name);

n_train=1:10;

scores=zeros(size(to_plot,1),length(n_train));


for i=1:size(to_plot,1)
    for t=1:length(n_train)
        if isnumeric(to_plot{i,1})
            idx1=data(:,1)==to_plot{i,1};
        else
            idx1=strcmp(str_data(2:end,1),to_plot{i,1});
        end
        idx=find(idx1 & data(:,2)==to_plot{i,2} & data(:,3)==to_plot{i,3} & data(:,4)==n_train(t));
        scores(i,t)=data(idx,5);
    end
end

figure('Position',[995   879   574   455])
plot(scores','MarkerSize',15,'Marker','.','LineWidth',2.1,'MarkerSize',19);
legend(to_plot(:,4),'Location','SouthEast','FontSize',11)
legend boxoff
ylim([0 100])
xlim([1 10])
xlabel('# Training Angles','FontSize',14)
ylabel('Classification Score (%)','FontSize',14)
%title('Classification Performance')
set(gca,'TickDir','out');
set(gca,'TickLength',[0.01 0.01]);
set(gca,'LineWidth',1.5,'FontWeight','Bold')
set(gcf,'Color','w');
box(gca,'off');
grid on


%saveas(gcf,[getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_model_' num2str(version) '.tif']);
%print([getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/Classification_by_model_' num2str(version) '_submission2'],'-dpdf');
%print([getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/actual_figures/Classification_by_model_all_runs_' num2str(version)],'-dpng');
export_fig([getDropboxDir 'Cox_Lab/Predictive_Networks/final_results/actual_figures/Classification_by_model_submission2.tif'])




end