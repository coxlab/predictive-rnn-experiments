function makePerturbationFigure


%faces_to_plot=[0:19];
faces_to_plot=[3,5,7,10,12,14];

feat_run_num=65;

angle_num=2;
clipset=10;
n_morphs=5;
pcs=[1,2,3];
pc_map=[1,6;2,16;3,12];


if feat_run_num==65
    is_GAN=0;
elseif feat_run_num==635
    is_GAN=1;
end

if is_GAN
    s_str='GAN_';
else
    s_str='';
end

if feat_run_num==67
    f_str='_epoch0';
else
    f_str='';
end


base_dir=['/home/bill/Data/FaceGen_Rotations/clipset' num2str(clipset) '/morphs/'];

orig_ims={};
all_ims={};
for i=1:length(pcs)
    all_ims{i}={};
    orig_ims{i}={};
    p=pcs(i);
    m=pc_map(pc_map(:,1)==p,2);
    feat_dir=['/home/bill/Projects/Predictive_Networks/facegen_' s_str 'runs_server/run_' num2str(feat_run_num) '/feature_analysis' f_str '/perturbation_analysis/clipset' num2str(clipset) '/mult_' num2str(m) '/'];
    feats=load([feat_dir 'perturbed_features.mat']);
    P=load([feat_dir 'perturbed_params.mat']);
    
    for j=1:length(faces_to_plot)
        f=faces_to_plot(j);
        
        idx=find(P.face_labels==f & P.angle_labels==angle_num*2);
        idx_start=(idx-1)*n_morphs+1;
        idx_end=idx*n_morphs;
        these_ims=feats.(['pca_' num2str(p)])(idx_start:idx_end,:,:);
        all_ims{i}{j}=reshape(these_ims,[n_morphs,150,150]);
        
        orig_ims{i}{j}={};
        for k=1:n_morphs
            im_file=[base_dir 'images_processed/face_' num2str(f) '_pc' num2str(p) '_p' num2str(k-1) '_angle' num2str(angle_num) '.png'];
            orig_ims{i}{j}{k}=im2double(imread(im_file));
        end
    end
    
    
end


val.dx1=100;
val.dx2=100;
for i=1:length(orig_ims)
    for j=1:length(orig_ims{i})
       for l=1:length(orig_ims{i}{j})
            for k=1:2
                v=min(orig_ims{i}{j}{l},[],k);
                rval=find(v<1,1,'first');
                lval=150-find(v<1,1,'last');
                val.(['dx' num2str(k)])=min(val.(['dx' num2str(k)]),rval);
                val.(['dx' num2str(k)])=min(val.(['dx' num2str(k)]),lval);
            end
      end
    end
end
dx=val.dx1;

for i=1:length(all_ims)
    for j=1:length(all_ims{i})
        all_ims{i}{j}=all_ims{i}{j}(:,:,dx-1:151-dx);
    end
end
for i=1:length(orig_ims)
    for j=1:length(orig_ims{i})
        for l=1:length(orig_ims{i}{j})
        orig_ims{i}{j}{l}=orig_ims{i}{j}{l}(:,dx-1:151-dx);
        end
    end
end
for i=1:length(faces_to_plot)
    %figure('Position',[91 20 1092 1311]);
    figure('Position',[86          15         782        1313]);
    h=tight_subplot(length(pcs)*2,n_morphs,[0.0 0.0],[0.00 0.00],[0.00 0.00]);
    disp(faces_to_plot(i))
    for j=1:length(pcs)
        for k=1:n_morphs
            ax_num=2*(j-1)*n_morphs+k;
            axes(h(ax_num));
            imshow(orig_ims{j}{i}{k});
            
            ax_num=2*(j-1)*n_morphs+k+n_morphs;
            axes(h(ax_num));
            imshow(squeeze(all_ims{j}{i}(k,:,:)));
        end
    end
    set(gcf,'Color','w')
    export_fig(['../final_results/perturb_plot_' s_str num2str(faces_to_plot(i)) '.tif'])
end
















end