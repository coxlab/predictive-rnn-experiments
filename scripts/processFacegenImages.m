function processFacegenImages

clipset=14;
desired_size=150;

base_dir=['/home/bill/Data/FaceGen_Rotations/clipset' num2str(clipset) '/'];

list=dir([base_dir 'images/']);

out_dir=[base_dir 'images_processed/'];
mkdir(out_dir);


for i=1:length(list)
    if ~isempty(strfind(list(i).name,'.png'))
        im=imread([base_dir 'images/' list(i).name]);
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
        im=imread([base_dir 'images/' list(i).name]);
        im=rgb2gray(im);
        im=im(shift:end-shift,shift:end-shift);
        im=imresize(im,desired_size/size(im,1));
        imwrite(im,[out_dir list(i).name]);
    end
end








end