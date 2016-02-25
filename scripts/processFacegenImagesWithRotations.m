function processFacegenImagesWithRotations

P.clipset=3;
P.desired_size=150;
P.initial_angle_range=[-pi/2,pi/2];
P.speed_range=[-pi/6,pi/6];

n_ims=25000+400;
n_frames=6;

initial_angles=zeros(n_ims,1);
angular_speeds=zeros(n_ims,1);

base_dir=['/home/bill/Data/FaceGen_Rotations/clipset' num2str(P.clipset) '/'];

rot_dir=[base_dir 'images_rotated/'];
mkdir(rot_dir);

for i=1:n_ims
    initial_angles(i)=P.initial_angle_range(1)+(P.initial_angle_range(2)-P.initial_angle_range(1))*rand;
    angular_speeds(i)=P.speed_range(1)+(P.speed_range(2)-P.speed_range(1))*rand;
    for j=1:n_frames
        theta=initial_angles(i)+(j-1)*angular_speeds(i);
        theta=180*theta/pi;
        f_name=[base_dir 'images/face_' num2str(i-1) '_frame_' num2str(j-1) '.png'];
        out_fname=[base_dir 'images_rotated/face_' num2str(i-1) '_frame_' num2str(j-1) '.png'];
        im=imread(f_name);
        im_rot=imrotate(im,theta,'bilinear','crop');
        Mrot = imrotate(ones(size(im)),theta,'bilinear','crop');
        idx=Mrot~=1;
        %im_rot(Mrot&~imclearborder(Mrot)) = im(1,1,1);
         im_rot(idx) = im(1,1,1);
        imwrite(im_rot,out_fname);
    end
end

save([base_dir 'postprocess_params.mat'],'P','initial_angles','angular_speeds');

list=dir(rot_dir);


out_dir=[base_dir 'images_processed/'];
mkdir(out_dir);


for i=1:length(list)
    if ~isempty(strfind(list(i).name,'.png'))
        im=imread([base_dir 'images_rotated/' list(i).name]);
        im=im2double(rgb2gray(im));
        x_max=min(im,[],1);
        y_max=min(im,[],2);
        if ~exist('hor_shift','var')
            hor_shift=find(x_max<1,1,'first');
            x_max=fliplr(x_max);
            idx2=find(x_max<1,1,'first');
            hor_shift=min(hor_shift,idx2);
            
            vert_shift=find(y_max<1,1,'first');
            y_max=flipud(y_max);
            idx2=find(y_max<1,1,'first');
            vert_shift=min(vert_shift,idx2);
        else
            idx1=find(x_max<1,1,'first');
            hor_shift=min(hor_shift,idx1);
            x_max=fliplr(x_max);
            idx2=find(x_max<1,1,'first');
            hor_shift=min(hor_shift,idx2);
            
            idx1=find(y_max<1,1,'first');
            vert_shift=min(vert_shift,idx1);
            y_max=flipud(y_max);
            idx2=find(y_max<1,1,'first');
            vert_shift=min(vert_shift,idx2);
        end      
    end
end
shift=min(vert_shift,hor_shift)-1;




for i=1:length(list)
    if ~isempty(strfind(list(i).name,'.png'))
        im=imread([base_dir 'images_rotated/' list(i).name]);
        im=rgb2gray(im);
        im=im(shift:end-shift,shift:end-shift);
        im=imresize(im,P.desired_size/size(im,1));
        imwrite(im,[out_dir list(i).name]);
    end
end








end