import pdb, sys, traceback, os, time, theano

import pickle as pkl
from prednet_GAN import create_feature_fxn, extract_model_features
from keras_models import initialize_model
def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *
cname = get_computer_name()
sys.path.append(get_scripts_dir() +'General_Scripts/')
import general_python_functions as gp
from general_python_functions import libsvm_classify
import hickle as hkl
import numpy as np
import sklearn as sk
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt
import sklearn.manifold as skm
import scipy.stats as ss
import prettyplotlib as ppl
import sklearn.decomposition as skd

sys.path.append('/home/bill/Libraries/keras/')
from keras.models import standardize_X, slice_X, make_batches

sys.path.append('/home/bill/Dropbox/Research/General_Python/')


base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_67/'

def prednet_pca_analysis():

    out_dir = base_run_dir+'analysis/pca_analysis/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    layer = 'RNN'
    output_num = 0
    t_steps = [0, 4]
    epochs = [0, 5, 25, 50]
    n_comp = 10

    curves = {}
    for t in t_steps:
        curves[t] = np.zeros((len(epochs),n_comp))

    for ei,e in enumerate(epochs):
        f_name = base_run_dir+'feature_analysis_epoch'+str(e)+'/features/'+layer+'_features.hkl'
        features = hkl.load(open(f_name,'r'))
        for t in t_steps:
            print str((t,e))
            X = features[output_num][:,t]
            pca = skd.PCA(n_components=n_comp)
            pca.fit(X)
            curves[t][ei] = 100*pca.explained_variance_ratio_


    plt.figure()
    for t in t_steps:
        plt.plot(curves[t].T)
        plt.legend(epochs, loc=0)
        plt.xlabel('pca component')
        plt.ylabel('percent explained variance')
        plt.title(layer+'_'+str(output_num)+' t='+str(t))
        plt.savefig(out_dir+'PCA_spectrum_'+layer+'_'+str(output_num)+'_t'+str(t)+'.jpg')
        plt.clf()


if __name__=='__main__':
	try:
		prednet_pca_analysis()

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
