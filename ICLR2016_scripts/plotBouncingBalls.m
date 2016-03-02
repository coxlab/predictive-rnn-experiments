function plotBouncingBalls

f_name=['/home/bill/Data/Bouncing_Balls/clip_set4/predictions/run_7/predictions.mat'];
load(f_name);

start_t=27;
plot_idx=1:3;
t_int=2;

n_plot=10;


for i=plot_idx
    figure;
    h=tight_subplot(2,n_plot,0.005,0.05,0.05);
   for j=1:n_plot 
        axes(h(j))
        imshow(squeeze(X(i,start_t+(j-1)*t_int,:,:)))
        axes(h(j+n_plot))
        imshow(squeeze(predictions(i,start_t+(j-1)*t_int,:,:)))
   end
end





end