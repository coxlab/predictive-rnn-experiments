function makeClassificationMDSPlot

for r=[65,67,138]
    d=load(['/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/classification_mds_run' num2str(r) '.mat']);
    data.(['run_' num2str(r)])=d.mds_mat;
    %data.(['run_' num2str(r)])=[d.mags',d.angles'];
end

labels=[];
for i=1:10
    labels=[labels,i*ones(1,12)];
end
for r=[65,67,138]
   figure
   scatter3(data.(['run_' num2str(r)])(:,1),data.(['run_' num2str(r)])(:,2),data.(['run_' num2str(r)])(:,3),100,labels,'filled')
   title(r)
    
    
end





end