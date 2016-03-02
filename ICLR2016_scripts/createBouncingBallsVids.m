function createBouncingBallsVids

load(['/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_training_set.mat']);
clips=data;

out_dir=[getDropboxDir 'Cox_Lab/Predictive_Networks/misc/example_bouncing_ball_clips/'];
mkdir(out_dir);
out_dir=[out_dir 'clip_set4/'];
mkdir(out_dir);

nmake=10;

for i=1:nmake
    writer=VideoWriter([out_dir 'clip_' num2str(i-1) '.avi']);
    open(writer);
    for j=1:size(clips,2)
        writeVideo(writer,squeeze(clips(i,j,:,:)));
    end
    close(writer)
end








end