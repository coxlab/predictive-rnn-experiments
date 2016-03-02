

m=6;
feat_dir=['/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/feature_analysis/perturbation_analysis/clipset5/'];
feats=load([feat_dir 'perturbed_features.mat']);

figure
for i=1:10
   starti=(i-1)*7+1;
   endi=i*7;
   count=1;
   for k=starti:endi
       subplot(1,10,count)
       imshow(squeeze(feats.predictions(k,1,:,:)));
       count=count+1;
   end 
    pause
    
    
    
end