function makePerturbPlots

%orig_angle_nums=[4,;
%morph_angle_nums=[4,5];
angle_nums=[4,5];
clipset=10;
n_morphs=5;

faces_to_plot=[0:19];
pcs=[1];

pc_map=[1,4;2,16;3,12];

base_dir=['/home/bill/Data/FaceGen_Rotations/clipset' num2str(clipset) '/morphs/'];


figure('Position',[96 242 1940 1092]);
h=tight_subplot(length(angle_nums)*3,n_morphs,[0.0 0.0],[0.00 0.00],[0.00 0.00]);
for p=pcs
    m=pc_map(pc_map(:,1)==p,2);
    feat_dir=['/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/feature_analysis/perturbation_analysis/clipset' num2str(clipset) '/mult_' num2str(m) '/'];
    feats.mse=load([feat_dir 'perturbed_features.mat']);
    %feat_dir=['/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_307/feature_analysis/perturbation_analysis/clipset' num2str(clipset) '/mult_' num2str(m) '/'];
    %feats.gan=load([feat_dir 'perturbed_features.mat']);
    P=load([feat_dir 'perturbed_params.mat']);
    for f=faces_to_plot
        for g={'mse'};
            all_ims.(g{1})={};
            for ai=1:length(angle_nums)
                idx=find(P.face_labels==f & P.angle_labels==angle_nums(ai)*2);
                idx_start=(idx-1)*n_morphs+1;
                idx_end=idx*n_morphs;
                these_ims=feats.(g{1}).(['pca_' num2str(p)])(idx_start:idx_end,:,:);
                all_ims.(g{1}){end+1,1}=reshape(these_ims,[n_morphs,150,150]);
            end
        end
        
        orig_ims={};
        for ai=1:length(angle_nums)
            orig_ims{ai}={};
            for i=1:n_morphs
                im_file=[base_dir 'images_processed/face_' num2str(f) '_pc' num2str(p) '_p' num2str(i-1) '_angle' num2str(angle_nums(ai)) '.png'];
                orig_ims{ai}{i}=imread(im_file);            
            end
        end
        
        for ai=1:length(angle_nums)
        for i=1:n_morphs
            axes(h(i+n_morphs*3*(ai-1)));
            imshow(orig_ims{ai}{i});
        end
        end
        
        for ai=1:length(angle_nums)
            for i=1:n_morphs
                axes(h(i+n_morphs*3*(ai-1)+n_morphs));
                imshow(squeeze(all_ims.mse{ai}(i,:,:)));
            end
        end
        
%         for ai=1:length(angle_nums)
%             for i=1:n_morphs
%                 axes(h(i+n_morphs*3*(ai-1)+2*n_morphs));
%                 imshow(squeeze(all_ims.gan{ai}(i,:,:)));
%             end
%         end
        
        disp(['pc ' num2str(p) ' face ' num2str(f)])
        pause
    end
    
end









end