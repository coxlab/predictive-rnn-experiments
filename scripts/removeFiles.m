function removeFiles

run_nums=171:253;

base_dir='/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/';

for r=run_nums
    delete([base_dir 'run_' num2str(r) '/*.hdf5'])
end



run_nums=92:201;

base_dir='/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/';

for r=run_nums
    delete([base_dir 'run_' num2str(r) '/*.hdf5'])
end



end