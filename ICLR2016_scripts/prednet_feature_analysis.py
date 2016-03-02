import pdb, sys, traceback, os, time, theano

import pickle as pkl
from prednet_GAN import create_feature_fxn, extract_model_features
from keras_models import initialize_model, initialize_GAN_models
def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *
cname = get_computer_name()
sys.path.append(get_scripts_dir() +'General_Scripts/')
import general_python_functions as gp
from misc_functions import readin_cv_index_file
from general_python_functions import libsvm_classify
import hickle as hkl
import numpy as np
import sklearn as sk
import sklearn.cross_validation as skcv
import sklearn.linear_model as sklm
import matplotlib.pyplot as plt
import sklearn.manifold as skm
import scipy.stats as ss
import sklearn.decomposition as skd
import prettyplotlib as ppl
import itertools
import pandas as pd
import scipy.io as spio
import scipy.stats as ss
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('/home/bill/Libraries/keras/')
from keras.models import standardize_X, slice_X, make_batches

sys.path.append('/home/bill/Dropbox/Research/General_Python/')
from tsne import tsne



def get_feature_params(param_overrides=None):

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_feature_runs/'
	P = {}
	P['is_GAN'] = False
	P['model_name'] = 'facegen_rotation_prednet_twolayer'
	P['model_params'] = {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
	P['weights_file'] =    '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/model_best_weights.hdf5' #'/home/bill/Projects/Predictive_Networks/facegen_runs/run_19/model_weights.hdf5'
	P['feature_layer_names'] = ['RNN']
	P['timesteps_to_use'] = [None]
	P['feature_is_time_expanded'] = [False]
	P['batch_size'] = P['model_params']['batch_size']
	P['data_file'] = '/home/bill/Data/FaceGen_Rotations/clipset4/clipstrain.hkl'
	P['calculate_idx'] = range(1000)
	P['input_nt'] = 5
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	P['D_model_name'] = 'facegen_rotation_prednet_twolayer_D'
	P['D_model_params'] = {'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False}

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P


def get_decoding_params(param_overrides=None):

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_feature_fit_runs/'
	P = {}
	P['feature_run_num'] = 4
	P['layers_to_use'] = ['RNN']
	P['layer_outputs_to_use'] = [range(5), [0]]
	P['timesteps_to_use'] = range(5)
	P['to_decode'] = ['pca_1']
	P['fit_model'] = 'ridge'
	P['model_params'] = [1e3,1e2,1,.1,1e-2]  #ordered from most regularization to least
	P['params_selection_method'] = 'best'
	P['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset4/all_params_train.pkl'
	P['n_splits'] = 5
	P['train_prop'] = 0.75
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P


def get_decoding_params_old(param_overrides=None):  #how old decoding params looked

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_feature_fit_runs/'
	P = {}
	P['feature_run_num'] = 4
	P['layers_to_use'] = ['RNN']
	P['layer_outputs_to_use'] = [range(5), [0]]
	P['timesteps_to_use'] = range(5)
	P['to_decode'] = ['pca_1']
	P['fit_model'] = 'ridge'
	P['model_params'] = [1e3,1e2,1,.1,1e-2]  #ordered from most regularization to least
	P['params_selection_method'] = 'best'
	P['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset4/all_params_train.pkl'
	P['n_splits'] = 5
	P['train_prop'] = 0.75
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P


def get_run_dir(run_num, is_server, is_GAN=False):

	if is_GAN:
		run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs'
	else:
		run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs'
	if is_server:
		run_dir = run_dir+'_server/'
	else:
		run_dir = run_dir+'/'
	run_dir = run_dir+'run_'+str(run_num)+'/'
	return run_dir


def run_run_full_decoding():

	extract_features = True
	fit_features = True
	is_GAN = False

	if is_GAN:
		run_num = 307
		is_server = True

		run_dir = get_run_dir(run_num, is_server, is_GAN)
		weights_file = run_dir+'G_model_weights_batch16500.hdf5'
		save_dir = run_dir + 'feature_analysis/'
		run_full_decoding_analysis(weights_file, save_dir, extract_features, fit_features, is_GAN)
	else:

		run_num = 65
		is_server = True

		run_dir = get_run_dir(run_num, is_server)

		if run_num==67:
			epochs_to_calc = range(151)

			for e in epochs_to_calc:
				print 'Running epoch: '+str(e)
				weights_file = run_dir+'model_weights_epoch'+str(e)+'.hdf5'
				save_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/'
				run_full_decoding_analysis(weights_file, save_dir, extract_features, fit_features, is_GAN)
		else:
			weights_file = run_dir+'model_best_weights.hdf5'
			save_dir = run_dir + 'feature_analysis/'
			run_full_decoding_analysis(weights_file, save_dir, extract_features, fit_features, is_GAN)

def get_run_map():

	run_dict = {}
	run_dict[65] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'MSE'}
	run_dict[67] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'MSE_150'}
	run_dict[139] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'AE_LSTM_dynamic'}
	run_dict[136] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'AE_LSTM_static'}
	run_dict[120] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_weights_epoch150', 'is_server': True, 'name': 'AE_LSTM_static_150'}
	run_dict[137] = {'is_GAN': False, 'is_autoencoder': True, 'model_param': 1024, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'AE_FC_units'}
	run_dict[138] = {'is_GAN': False, 'is_autoencoder': True, 'model_param': 4096, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'AE_FC_weights'}
	run_dict[342] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': False, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': False, 'name': 'GAN_shared_RNN'}
	run_dict[499] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_1024fcpre'}
	run_dict[506] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_1024fcpre'}
	run_dict[507] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_1024fcpre'}
	run_dict[509] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_1024fcpre_mult2'}
	run_dict[511] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_1024fcpre_mult2'}
	run_dict[524] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True, 'fc_precat_size': 512}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_512'}
	run_dict[529] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': True, 'fc_precat_size': 256}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_shared_RNN_25'}
	run_dict[544] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_not_sharedRNN'}
	run_dict[573] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_late_fusion'}
	run_dict[582] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_initialized'}
	run_dict[635] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_weights', 'is_server': True, 'name': 'GAN_initialized'}
	run_dict[668] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_weights_batch1000', 'is_server': True, 'name': 'GAN_submission2'}
	run_dict[635] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_submission2_v2'}
	run_dict[662] = {'is_GAN': True, 'is_autoencoder': False, 'model_param': {'rand_max': 0.5, 'use_fc_precat': True, 'share_RNN': False, 'fc_precat_size': 1024}, 'weights_file': 'G_model_best_weights', 'is_server': True, 'name': 'GAN_submission2_v3'}
	run_dict[144] = {'is_GAN': False, 'is_autoencoder': False, 'model_param': None, 'weights_file': 'model_best_weights', 'is_server': True, 'name': 'AE_LSTM_dynamic_full'}

	return run_dict



def run_run_full_decoding2(run_num, extract_features=True, fit_features=True, static=False):

	run_dict = get_run_map()
	run_dir = get_run_dir(run_num, run_dict[run_num]['is_server'], run_dict[run_num]['is_GAN'])
	weights_file = run_dir + run_dict[run_num]['weights_file']+'.hdf5'
	save_dir = run_dir + 'feature_analysis/'

	if static:
		clipset = 18
	else:
		clipset = 5

	run_full_decoding_analysis(weights_file, save_dir, extract_features=extract_features, fit_features=fit_features, is_GAN=run_dict[run_num]['is_GAN'], is_autoencoder=run_dict[run_num]['is_autoencoder'], model_param=run_dict[run_num]['model_param'], clipset=clipset)


def run_full_decoding_analysis(weights_file, save_dir, extract_features=True, fit_features=True, is_GAN=False, is_autoencoder=False, model_param=None, clipset=5):

	if clipset==5:
		s_str = ''
	elif clipset==18:
		s_str = '_static'

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if extract_features:
		po = {}
		po['is_GAN'] = is_GAN
		po['is_autoencoder'] = is_autoencoder
		if is_autoencoder:
			po['model_name'] = 'facegen_rotation_autoencoder' #'facegen_rotation_prednet_twolayer'
			po['model_params'] = {'n_FC': model_param, 'batch_size': 4, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
		elif is_GAN:
			po['model_name'] = 'facegen_rotation_prednet_twolayer_G_to_D'
			if 'use_fc_precat' in model_param:
				use_fc = model_param['use_fc_precat']
			else:
				use_fc = False
			if 'fc_precat_size' in model_param:
				fc_size = model_param['fc_precat_size']
			else:
				fc_size = 1024
			po['model_params'] = {'n_timesteps': 5, 'batch_size': 4, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True}
			po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': model_param['share_RNN'], 'RNN_params_fixed': False, 'use_fc_precat': use_fc, 'fc_precat_size': fc_size}
		else:
			po['model_name'] = 'facegen_rotation_prednet_twolayer'
			if '144' in weights_file:
				nt = 6
			else:
				nt = 5
			po['model_params'] = {'n_timesteps': nt, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
		po['weights_file'] = weights_file
		if po['is_autoencoder']:
			po['feature_layer_names'] = ['fc_encoder_relu'] #, 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [0] #, None, None]
			po['feature_is_time_expanded'] = [False] #, False, False]
		else:
			po['feature_layer_names'] = ['RNN'] #['RNN'] #, 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [None] #, None, None]
			po['feature_is_time_expanded'] = [False]
		po['batch_size'] = po['model_params']['batch_size']
		po['data_file'] = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/clipsall.hkl'
		po['calculate_idx'] = range(4000)
		if '144' in weights_file:
			po['input_nt'] = 6
		else:
			po['input_nt'] = 5
		po['run_num'] = -1
		po['save_dir'] = save_dir+'features'+s_str+'/'

		run_extract_features(po)

	if fit_features:

		po = {}

		if is_autoencoder:
			po['layers_to_use'] = ['fc_encoder_relu']
			po['timesteps_to_use'] = [[0]]
			po['layer_outputs_to_use'] = [[0]]
		else:
			po['layers_to_use'] = ['RNN'] #['pool0', 'pool1', 'RNN', 'fc_decoder_relu', 'deconv1_relu']
			if '144' in weights_file:
				po['timesteps_to_use'] = [[5]]
			else:
				po['timesteps_to_use'] = [[4]]
			po['layer_outputs_to_use'] = [[0]]
		to_decode = ['pan_angular_speeds', 'pan_initial_angles']
		for r in range(50):
			to_decode.append('pca_'+str(r))
		po['to_decode'] = to_decode #['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pca_6', 'pca_7', 'pca_8', 'pca_9', 'pca_10', 'pca_11', 'pca_12', 'pca_13', 'pca_14', 'pca_15', 'pca_16', 'pan_angular_speeds', 'pan_initial_angles'] #['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pan_angles', 'pan_angular_speeds', 'pan_initial_angles','pan_angles_linear']
		po['fit_model'] = 'ridge'
		po['model_params'] = {'method': 'adaptive', 'start_list': [1e3,1e2,1,.1,1e-2], 'max_param': 1e5, 'min_param': 1e-5}  #ordered from most regularization to least
		po['ntrain'] = 2000
		po['nval'] = 1000
		po['ntest'] = 1000
		po['params_selection_method'] = 'best'
		po['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/all_params_all.pkl'
		po['run_num'] = -1
		po['save_dir'] = save_dir+'decoding'+s_str+'/'

		f = open(save_dir+'features/params.pkl','r')
		P_feature = pkl.load(f)
		f.close()

		run_fit_features(param_overrides=po, P_feature=P_feature)


def run_run_pca_decoding():

	run_num = 67
	is_server = True
	epochs_to_calc = [0,5,10,25,50]

	run_dir = get_run_dir(run_num, is_server)

	for e in epochs_to_calc:
		print 'Running epoch: '+str(e)
		weights_file = run_dir+'model_weights_epoch'+str(e)+'.hdf5'
		save_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/'
		run_pca_decoding_analysis(weights_file, save_dir)


def run_pca_decoding_analysis(weights_file, save_dir):

	po = {}
	po['layers_to_use'] = ['RNN'] #['pool0', 'pool1', 'RNN', 'fc_decoder_relu', 'deconv1_relu']
	po['layer_outputs_to_use'] = [[0,1]] #[[0], [0], range(5), [0], [0]]
	po['timesteps_to_use'] = [[0,4]]  #[[0], [0], range(5), [-1], [-1]]
	po['to_decode'] = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pan_angular_speeds', 'pan_angles']
	po['fit_model'] = 'ridge'
	po['model_params'] = {'method': 'adaptive', 'start_list': [1e3,1e2,1,.1,1e-2], 'max_param': 1e5, 'min_param': 1e-8}  #ordered from most regularization to least
	po['ntrain'] = 2000
	po['nval'] = 1000
	po['ntest'] = 1000
	po['params_selection_method'] = 'best'
	po['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
	po['run_num'] = -1
	base_save_dir = save_dir+'pca_decoding/'

	f = open(save_dir+'features/params.pkl','r')
	P_feature = pkl.load(f)
	f.close()

	if not os.path.exists(base_save_dir):
		os.mkdir(base_save_dir)

	po['use_pca'] = True

	idxs = [range(1), range(5), range(10), range(25), range(100)]

	for i in idxs:
		po['pca_idx'] = i
		s = get_pca_idx_str(i)
		po['save_dir'] = base_save_dir + s+'/'
		run_fit_features(param_overrides=po, P_feature=P_feature)


def run_run_classification_decoding(run_num, version):

	if run_num==67:
		is_GAN = False
		is_autoencoder = False
		epochs_to_calc = [0,10,25,50,150] #[0,5,10,25,50,100] #range(147,151)
		model_param = None
	elif run_num==98 or run_num==65 or run_num==120 or run_num==139 or run_num==136:
		is_GAN = False
		is_autoencoder = False
		epochs_to_calc = None
		model_param = None
	elif run_num==110 or run_num==133 or run_num==134 or run_num==135 or run_num==137 or run_num==138 or run_num==142 or run_num==143:
		is_GAN = False
		is_autoencoder = True
		epochs_to_calc = None
		if run_num in [110, 133, 137]:
			model_param = 1024
		elif run_num in [134, 135, 138, 142, 143]:
			model_param = 4096
	elif run_num==307 or run_num==276 or run_num==277 or run_num==452 or run_num==322 or run_num==332 or run_num==342 or run_num==478 or run_num in [499,511,573,668,662,635]:
		is_GAN = True
		is_autoencoder = False
		epochs_to_calc = None
		if run_num==452 or run_num==322 or run_num==342 or run_num in [478,573,668,662,635]:
			rand_max = 0.5
		else:
			rand_max = 0.1
		model_param = {}
		model_param['rand_max'] = rand_max
		if run_num in [499,506,507,509,511,573,668]:
			model_param['use_fc_precat'] = True
		else:
			model_param['use_fc_precat'] = False
		model_param['share_RNN'] = True   #shouldnt make a difference


	if run_num==322 or run_num==332 or run_num==342:
		is_server = False
	else:
		is_server = True
	extract_features = True
	fit_features = True


	run_dir = get_run_dir(run_num, is_server, is_GAN)

	if epochs_to_calc is not None:
		for e in epochs_to_calc:
			print 'Running epoch: '+str(e)
			weights_file = run_dir+'model_weights_epoch'+str(e)+'.hdf5'
			save_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/classification/version_'+str(version)+'/'
			run_classification_decoding(weights_file, save_dir, version, extract_features=extract_features, fit_features=fit_features, is_GAN=is_GAN, is_autoencoder=is_autoencoder, model_param=model_param)
	else:
		if is_GAN and run_num==307:
			weights_file = run_dir+'G_model_weights_batch16500.hdf5'
		elif is_GAN and run_num==668:
			weights_file = run_dir+'G_model_weights_batch1000.hdf5'
		elif is_GAN:
			weights_file = run_dir+'G_model_best_weights.hdf5'
		else:
			weights_file = run_dir+'model_best_weights.hdf5'
		save_dir = run_dir + 'feature_analysis/classification/version_'+str(version)+'/'
		run_classification_decoding(weights_file, save_dir, version, extract_features=extract_features, fit_features=fit_features, is_GAN=is_GAN, is_autoencoder=is_autoencoder, model_param=model_param)


def create_classification_cv_file(version):

	if version==0:
		f_name = '/home/bill/Data/FaceGen_Rotations/clipset8/params.pkl'
		params_dict = pkl.load(open(f_name, 'r'))
		cv_num = 0
		d = {'face_labels': (6,2,2), 'pan_speed_labels': (300,100,100)}

		for v in ['face_labels', 'pan_speed_labels']:
			train_idx = []
			val_idx = []
			test_idx = []
			u_classes = np.unique(params_dict[v])
			for c in u_classes:
				idx = np.nonzero(params_dict[v]==c)[0]
				idx = np.random.permutation(idx).tolist()
				train_idx.extend(idx[:d[v][0]])
				val_idx.extend(idx[d[v][0]:d[v][0]+d[v][1]])
				test_idx.extend(idx[d[v][0]+d[v][1]:])
			f_name = '/home/bill/Data/FaceGen_Rotations/clipset8/cv_idx_'+v+'_0.pkl'
			pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))
	elif version==1:
		n_use = 100
		for i in range(5):
			train_idx = np.arange(i, 5*n_use, 5)
			#poss_val = list(set(range(5))-set([i]))
			#val_num = np.random.permutation(poss_val)[0]
			#val_idx = list(set(np.arange(400, 500))-set(np.arange(400+i, 500, 5)))
			#val_idx = np.arange(val_num, 5*n_use, 5)
			val_idx = []
			test_idx = [j for j in range(5*n_use) if (j not in train_idx and j not in val_idx)]
			f_name = '/home/bill/Data/FaceGen_Rotations/clipset9/cv_idx_'+str(i)+'.pkl'
			pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))
	elif version==2:
		n_cv = 40

		out_dir = '/home/bill/Data/FaceGen_Rotations/clipset10/cv_idxs/version_2/'
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

		f_name = '/home/bill/Data/FaceGen_Rotations/clipset10/params.pkl'
		params_dict = pkl.load(open(f_name, 'r'))
		labels = params_dict['pan_angle_labels']
		n_angles = params_dict['P']['n_angles']
		n_faces = params_dict['P']['n_faces']
		n_ex = n_angles*n_faces
		for n_train in range(1, n_angles-1):
			all_pairs = [tup for tup in itertools.combinations(range(n_angles),n_train)]
			if len(all_pairs)>n_cv:
				all_pairs = np.random.permutation(all_pairs)[:n_cv]
			for k,tup in enumerate(all_pairs):
				train_nums = list(tup)
				poss_val = list(set(range(n_angles))-set(train_nums))
				val_num = np.random.permutation(poss_val)[0]
				test_nums = [i for i in range(n_angles) if (i not in train_nums and i!=val_num)]
				train_idx = [i for i in range(n_ex) if labels[i] in train_nums]
				val_idx = [i for i in range(n_ex) if labels[i]==val_num]
				test_idx = [i for i in range(n_ex) if labels[i] in test_nums]
				f_name = out_dir+'cv_idx_ntrain'+str(n_train)+'_'+str(k)+'.pkl'
				pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))
	elif version==3:
		n_cv = 40

		out_dir = '/home/bill/Data/FaceGen_Rotations/clipset10/cv_idxs/version_3/'
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)

		f_name = '/home/bill/Data/FaceGen_Rotations/clipset10/params.pkl'
		params_dict = pkl.load(open(f_name, 'r'))
		labels = params_dict['face_labels']
		n_angles = params_dict['P']['n_angles']
		n_faces = params_dict['P']['n_faces']
		n_ex = n_angles*n_faces
		for n_train in range(1, 11):
			all_pairs = []
			while len(all_pairs)<n_cv:
				idx = np.random.permutation(n_faces)
				tup = itertools.combinations(idx,n_train).next()
				if tup not in all_pairs:
					all_pairs.append(tup)
			for k,tup in enumerate(all_pairs):
				train_nums = list(tup)
				poss_val = list(set(range(n_faces))-set(train_nums))
				val_nums = np.random.permutation(poss_val)[:4]
				test_nums = [i for i in range(n_faces) if (i not in train_nums and i not in val_nums)]
				train_idx = [i for i in range(n_ex) if labels[i] in train_nums]
				val_idx = [i for i in range(n_ex) if labels[i] in val_nums]
				test_idx = [i for i in range(n_ex) if labels[i] in test_nums]
				f_name = out_dir+'cv_idx_ntrain'+str(n_train)+'_'+str(k)+'.pkl'
				pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))
	elif version==4:
		n_cv = 40

		out_dir = '/home/bill/Data/FaceGen_Rotations/clipset11/cv_idxs/version_4/'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		f_name = '/home/bill/Data/FaceGen_Rotations/clipset11/params.pkl'
		params_dict = pkl.load(open(f_name, 'r'))
		labels = params_dict['face_labels']
		n_angles = params_dict['P']['n_angles']
		n_faces = params_dict['P']['n_faces']
		n_ex = n_angles*n_faces
		for n_train in range(1, n_faces-1):
			all_pairs = [tup for tup in itertools.combinations(range(n_faces),n_train)]
			if len(all_pairs)>n_cv:
				all_pairs = np.random.permutation(all_pairs)[:n_cv]
			for k,tup in enumerate(all_pairs):
				train_nums = list(tup)
				poss_val = list(set(range(n_faces))-set(train_nums))
				val_num = np.random.permutation(poss_val)[0]
				test_nums = [i for i in range(n_faces) if (i not in train_nums and i!=val_num)]
				train_idx = [i for i in range(n_ex) if labels[i] in train_nums]
				val_idx = [i for i in range(n_ex) if labels[i]==val_num]
				test_idx = [i for i in range(n_ex) if labels[i] in test_nums]
				f_name = out_dir+'cv_idx_ntrain'+str(n_train)+'_'+str(k)+'.pkl'
				pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))
	elif version>4:
		if version==5:
			clipset = 14
			n_cv = 40
			invariant_var = 'pan_angle_labels'

		out_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/cv_idxs/version_'+str(version)+'/'
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		f_name = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/params.pkl'
		params_dict = pkl.load(open(f_name, 'r'))
		labels = params_dict[invariant_var]
		n_angles = params_dict['P']['n_angles']
		n_faces = params_dict['P']['n_faces']
		if invariant_var=='pan_angle_labels':
			n_invar = n_angles
		else:
			n_invar = n_faces
		n_ex = n_angles*n_faces
		for n_train in range(1, n_invar-1):

			all_pairs = [tup for tup in itertools.combinations(range(n_invar),n_train)]
			if len(all_pairs)>n_cv:
				all_pairs = np.random.permutation(all_pairs)[:n_cv]
			# all_pairs = []
			# while len(all_pairs)<n_cv:
			# 	idx = np.random.permutation(n_invar)
			# 	tup = itertools.combinations(idx,n_train).next()
			# 	if tup not in all_pairs:
			# 		all_pairs.append(tup)
			for k,tup in enumerate(all_pairs):
				train_nums = list(tup)
				poss_val = list(set(range(n_invar))-set(train_nums))
				val_num = np.random.permutation(poss_val)[0]
				test_nums = [i for i in range(n_invar) if (i not in train_nums and i!=val_num)]
				train_idx = [i for i in range(n_ex) if labels[i] in train_nums]
				val_idx = [i for i in range(n_ex) if labels[i]==val_num]
				test_idx = [i for i in range(n_ex) if labels[i] in test_nums]
				f_name = out_dir+'cv_idx_ntrain'+str(n_train)+'_'+str(k)+'.pkl'
				pkl.dump([train_idx,val_idx,test_idx], open(f_name,'w'))



def compute_classification_MDS():

	run_nums = [65, 67, 138]
	version = 2
	n_faces = 10

	for r in run_nums:

		run_dir = get_run_dir(r, True, False)

		if r==67:
			base_dir = run_dir + 'feature_analysis_epoch0/classification/version_'+str(version)+'/'
		else:
			base_dir = run_dir + 'feature_analysis/classification/version_'+str(version)+'/'

		if r==138:
			f_name = 'fc_encoder_relu'
		else:
			f_name = 'RNN'

		if r==65:
			t = 4
		else:
			t = 0

		features = hkl.load(open(base_dir+'features/'+f_name+'_features.hkl','r'))
		features = features[0][:,t]
		features = features[0:12*n_faces].astype(np.float64)
		clf = skm.MDS(n_components=3)
		X_mds = clf.fit_transform(features)
		#X_tsne = tsne(features)
		# angles = np.zeros(features.shape[0])
		# mags = np.zeros(features.shape[0])
		# v = np.ones(features.shape[1])
		# for i in range(features.shape[0]):
		# 	mags[i] = np.linalg.norm(features[i])
		# 	angles[i] = np.arccos( np.dot(features[i],v)/(mags[i]*np.linalg.norm(v) ))

		save_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/classification_mds_run'+str(r)+'.mat'
		spio.savemat(save_dir, {'mds_mat': X_mds})
		#save_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/classification_norms_run'+str(r)+'.mat'
		#spio.savemat(save_dir, {'mags': mags, 'angles': angles})


def run_random_classification_decoding():

	n_runs = 10
	extract_features = True

	base_dir = '/home/bill/Projects/Predictive_Networks/facegen_random_weights/'

	if extract_features:
		for i in range(n_runs):
			weights_file = '/home/bill/Projects/Predictive_Networks/models/facegen_initializations/weights_'+str(i)+'.hdf5'

			save_dir = base_dir + 'run_'+str(i)+'/'
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)

			po = {}
			po['model_name'] = 'facegen_rotation_prednet_twolayer'
			po['model_params'] = {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
			po['weights_file'] = weights_file
			po['feature_layer_names'] = ['RNN'] #, 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [None] #, None, None]
			po['feature_is_time_expanded'] = [False] #, False, False]
			po['batch_size'] = po['model_params']['batch_size']
			po['data_file'] = '/home/bill/Data/FaceGen_Rotations/clipset8/clipsall.hkl'
			po['calculate_idx'] = range(5000)
			po['input_nt'] = 5
			po['run_num'] = -1
			po['save_dir'] = save_dir+'features/'

			run_extract_features(po)

	if fit_features:
		for i in range(n_runs):
			save_dir = base_dir + 'run_'+str(i)+'/'

			po = {}
			po['layers_to_use'] = ['RNN']
			po['layer_outputs_to_use'] = [[0,1]]
			po['timesteps_to_use'] = [[0,2,4]]
			po['to_decode'] = ['face_labels']
			po['fit_model'] = 'svm'
			po['model_params'] = {'method': 'adaptive', 'start_list': [1e-2,1,1e2,1e4,1e5], 'max_param': 1e5, 'min_param': 1e-4}  #ordered from most regularization to least
			po['cv_file'] = ['/home/bill/Data/FaceGen_Rotations/clipset8/cv_idx_face_labels_0.pkl','/home/bill/Data/FaceGen_Rotations/clipset8/cv_idx_pan_speed_labels_0.pkl']
			po['params_selection_method'] = 'best'
			po['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset8/params.pkl'
			po['run_num'] = -1
			po['save_dir'] = save_dir+'decoding/'

			f = open(save_dir+'features/params.pkl','r')
			P_feature = pkl.load(f)
			f.close()

			run_fit_features(param_overrides=po, P_feature=P_feature)

def analyze_random_classification():

	base_dir = '/home/bill/Projects/Predictive_Networks/facegen_random_weights/'
	n_runs = 10
	layers = [0,1]
	timesteps = [0,2,4]

	scores = {}
	for l in layers:
		for t in timesteps:
			scores[(l,t)] = np.zeros(n_runs)

	for i in range(n_runs):
		d = base_dir + 'run_'+str(i)+'/'
		s = pkl.load(open(d+'decoding/test_scores.pkl', 'r'))
		for l in layers:
			for t in timesteps:
				scores[(l,t)][i] = s[('RNN',l,t,'face_labels')]

	for l in layers:
		for t in timesteps:
			print 'For output_num='+str(l)+', t='+str(t)+': '+str(np.mean(scores[(l,t)]))+'+-'+str(np.std(scores[(l,t)]))



def run_classification_decoding(weights_file, save_dir, version, extract_features=True, fit_features=True, is_GAN=False, is_autoencoder=False, model_param=None):

	if version==0:
		#500 faces with 10 rotation speeds
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset8/clipsall.hkl'
		calc_idx = range(5000)
		to_decode = ['face_labels','pan_speed_labels']
		cv_files = ['/home/bill/Data/FaceGen_Rotations/clipset8/cv_idx_face_labels_0.pkl','/home/bill/Data/FaceGen_Rotations/clipset8/cv_idx_pan_speed_labels_0.pkl']
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset8/params.pkl'
		fit_timesteps = [[0,2,4]]
	elif version==1:
		# 100 faces with 5 diff angles
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset9/clipsall.hkl'
		calc_idx = range(500)
		to_decode = ['face_labels']
		cv_files = ['/home/bill/Data/FaceGen_Rotations/clipset9/cv_idx_'+str(i)+'.pkl' for i in range(5)]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset9/params.pkl'
		fit_timesteps = [[4]]
	elif version==2:
		# 50 faces with 12 diff angles
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset10/clipsall.hkl'
		calc_idx = range(600)
		to_decode = ['face_labels']
		cv_dir = '/home/bill/Data/FaceGen_Rotations/clipset10/cv_idxs/version_2/'
		cv_files = os.walk(cv_dir).next()[2]
		cv_files = [cv_dir + c for c in cv_files]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset10/params.pkl'
		fit_timesteps = [[0,4]]
	elif version==3:
		# 50 faces with 12 diff angles
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset10/clipsall.hkl'
		calc_idx = range(600)
		to_decode = ['pan_angle_labels']
		cv_dir = '/home/bill/Data/FaceGen_Rotations/clipset10/cv_idxs/version_3/'
		cv_files = os.walk(cv_dir).next()[2]
		cv_files = [cv_dir + c for c in cv_files]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset10/params.pkl'
		fit_timesteps = [[0,4]]
	elif version==4:
		# 12 faces with 50 diff angles
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset11/clipsall.hkl'
		calc_idx = range(600)
		to_decode = ['pan_angle_labels']
		cv_dir = '/home/bill/Data/FaceGen_Rotations/clipset11/cv_idxs/version_4/'
		cv_files = os.walk(cv_dir).next()[2]
		cv_files = [cv_dir + c for c in cv_files]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset11/params.pkl'
		fit_timesteps = [[0,4]]
	elif version>4:
		if version==5:
			clipset = 14
			n_ex = 500*12
			decode_var = 'face_labels'

		data_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/clipsall.hkl'
		calc_idx = range(n_ex)
		to_decode = [decode_var]
		cv_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/cv_idxs/version_5/'
		cv_files = os.walk(cv_dir).next()[2]
		cv_files = [cv_dir + c for c in cv_files]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/params.pkl'
		fit_timesteps = [[0,4]]

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if extract_features:

		po = {}
		po['is_GAN'] = is_GAN
		po['is_autoencoder'] = is_autoencoder
		if is_autoencoder:
			po['model_name'] = 'facegen_rotation_autoencoder' #'facegen_rotation_prednet_twolayer'
			po['model_params'] = {'n_FC': model_param, 'batch_size': 4, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
		elif is_GAN:
			po['model_name'] = 'facegen_rotation_prednet_twolayer_G_to_D'
			po['model_params'] = {'n_timesteps': 5, 'batch_size': 4, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True}
			if 'use_fc_precat' in model_param:
				use_fc = model_param['use_fc_precat']
			else:
				use_fc = False
			po['rand_max'] = model_param['rand_max']
			po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': model_param['share_RNN'], 'RNN_params_fixed': False, 'use_fc_precat': use_fc, 'fc_precat_size': 1024}
		else:
			po['model_name'] = 'facegen_rotation_prednet_twolayer'
			po['model_params'] = {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
		po['weights_file'] = weights_file
		if po['is_autoencoder']:
			po['feature_layer_names'] = ['fc_encoder_relu'] #, 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [0] #, None, None]
			po['feature_is_time_expanded'] = [False] #, False, False]
		else:
			po['feature_layer_names'] = ['RNN'] #['RNN'] #, 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [None] #, None, None]
			po['feature_is_time_expanded'] = [False] #, False, False]
		po['batch_size'] = po['model_params']['batch_size']
		po['data_file'] = data_file
		po['calculate_idx'] = calc_idx
		po['input_nt'] = 5
		po['run_num'] = -1
		po['save_dir'] = save_dir+'features/'

		run_extract_features(po)

	if fit_features:

		po = {}
		po['layer_outputs_to_use'] = [[0]] #[[0], [0], range(5), [0], [0]]
		if is_autoencoder:
			po['layers_to_use'] = ['fc_encoder_relu']
			po['timesteps_to_use'] = [[0]]
		else:
			po['layers_to_use'] = ['RNN'] #['pool0', 'pool1', 'RNN', 'fc_decoder_relu', 'deconv1_relu']
			po['timesteps_to_use'] = [[0,4]]  #[[0], [0], range(5), [-1], [-1]]
		po['to_decode'] = to_decode
		po['fit_model'] = 'svm'
		po['model_params'] = {'method': 'adaptive', 'start_list': [1,1e2,1e4], 'max_param': 1e5, 'min_param': 1e-4}  #ordered from most regularization to least
		po['params_selection_method'] = 'best'
		po['params_file'] = params_file
		po['run_num'] = -1

		f = open(save_dir+'features/params.pkl','r')
		P_feature = pkl.load(f)
		f.close()

		if version>0:
			for i,cv in enumerate(cv_files):
				po['cv_file'] = [cv]
				idx0 = cv.rfind('/')
				idx1 = cv.rfind('.pkl')
				s = cv[idx0+1:idx1]
				po['save_dir'] = save_dir+'decoding/'+s+'/'
				if not os.path.exists(po['save_dir']):
					os.makedirs(po['save_dir'])
				run_fit_features(param_overrides=po, P_feature=P_feature)
		else:
			po['cv_file'] = cv_files
			po['save_dir'] = save_dir+'decoding/'
			run_fit_features(param_overrides=po, P_feature=P_feature)


def run_feature_classification(feature, extract_features=True, fit_features=True):

	version=2
	if version==2:
		data_file = '/home/bill/Data/FaceGen_Rotations/clipset10/clipsall.hkl'
		calc_idx = range(600)
		to_decode = ['face_labels']
		cv_dir = '/home/bill/Data/FaceGen_Rotations/clipset10/cv_idxs/version_2/'
		cv_files = os.walk(cv_dir).next()[2]
		cv_files = [cv_dir + c for c in cv_files]
		params_file = '/home/bill/Data/FaceGen_Rotations/clipset10/params.pkl'

	save_dir = '/home/bill/Projects/Predictive_Networks/classification_results/version_2/'+feature+'/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if extract_features:
		X = hkl.load(open(data_file, 'r'))
		X = X[calc_idx]
		X = X[:,0]

		if feature=='pixels':
		 	features = [X.reshape((len(calc_idx), 1, X.shape[-1]*X.shape[-2]))]
		elif feature=='lbp':
			from skimage.feature import local_binary_pattern
			feats = np.zeros(X.shape)
			for i in range(feats.shape[0]):
				feats[i,0] = local_binary_pattern(X[i,0], 24, 3)
			features = [feats.reshape(feats.shape[0], 1, feats.shape[2]*feats.shape[3])]
		elif feature=='hog':
			from skimage.feature import hog
			for i in range(X.shape[0]):
				tmp = hog(X[i,0])
				if i==0:
					feats = np.zeros((X.shape[0], 1, len(tmp)))
				feats[i,0] = tmp
			features = [feats]

		out_dir = save_dir+'features/'
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		feat_file = out_dir+feature+'_features.hkl'
		hkl.dump(features, open(feat_file,'w'))


	if fit_features:
		P_feature = {}
		P_feature['save_dir'] = save_dir+'features/'
		P_feature['calculate_idx'] = calc_idx

		po = {}
		po['layer_outputs_to_use'] = [[0]]
		po['layers_to_use'] = [feature]
		po['timesteps_to_use'] = [[0]]
		po['to_decode'] = to_decode
		po['fit_model'] = 'svm'
		po['model_params'] = {'method': 'adaptive', 'start_list': [1,1e2,1e4], 'max_param': 1e5, 'min_param': 1e-4}  #ordered from most regularization to least
		po['params_selection_method'] = 'best'
		po['params_file'] = params_file
		po['run_num'] = -1

		for i,cv in enumerate(cv_files):
			po['cv_file'] = [cv]
			idx0 = cv.rfind('/')
			idx1 = cv.rfind('.pkl')
			s = cv[idx0+1:idx1]
			po['save_dir'] = save_dir+'decoding/'+s+'/'
			if not os.path.exists(po['save_dir']):
				os.makedirs(po['save_dir'])
			run_fit_features(param_overrides=po, P_feature=P_feature)



def aggregate_classification_results(version):

	if version==2 or version==5:
		var = 'face_labels'
	elif version==3 or version==4:
		var = 'pan_angle_labels'

	b_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'

	if False:
		for r in [143]: #[65,98,110,120,133,134,135,139,136,137,138]:

			base_dir = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/decoding/'
			out_dir = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/'
			if r in [110,133,134,135,137,138,142,143]:
				scores = get_classification_results_for_folder(base_dir, var, layer='fc_encoder_relu', timesteps=[0])
			else:
				scores = get_classification_results_for_folder(base_dir,var)
			pkl.dump(scores, open(out_dir+'scores_summary.pkl','w'))

	if False:
		for epoch in [0,10,25,50,150]:
			base_dir = b_dir+'run_67/feature_analysis_epoch'+str(epoch)+'/classification/version_'+str(version)+'/decoding/'
			out_dir = b_dir+'run_67/feature_analysis_epoch'+str(epoch)+'/classification/version_'+str(version)+'/'
			scores = get_classification_results_for_folder(base_dir,var)
			pkl.dump(scores, open(out_dir+'scores_summary.pkl','w'))

	b_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/'
	#b_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'

	if True:
		for r in [635]: #[307,276,277,452]:

			base_dir = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/decoding/'
			out_dir = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/'
			scores = get_classification_results_for_folder(base_dir,var)
			pkl.dump(scores, open(out_dir+'scores_summary.pkl','w'))

	if False:
		for f in ['pixels', 'lbp', 'hog']:
			base_dir = '/home/bill/Projects/Predictive_Networks/classification_results/version_'+str(version)+'/'+f+'/'
			scores = get_classification_results_for_folder(base_dir+'decoding/',var, layer=f, timesteps=[0])
			pkl.dump(scores, open(base_dir+'scores_summary.pkl','w'))


def aggregate_classification_results2():

	version = 2
	var = 'face_labels'
	layer = 'fc_decoder_relu'
	timesteps = [-1]

	b_dir = {}
	b_dir[65] = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
	b_dir[452] = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/'

	for r in [65, 452]:
		base_dir = b_dir[r]+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/decoding/'
		out_dir = b_dir[r]+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/'
		scores = get_classification_results_for_folder(base_dir,var)
		pkl.dump(scores, open(out_dir+'scores_summary_' +layer+'.pkl','w'))


def aggregate_decoding_results():

	var = ['pan_angular_speeds', 'pan_angles', 'pan_angles_linear', 'pan_initial_angles', 'pca_1', 'pca_2', 'pca_3', 'pca_4','pca_5','pca_6','pca_7','pca_8','pca_9', 'pca_10']
	epochs = range(151)
	timesteps = [0,2,4]
	output_nums = [0,1]

	b_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
	out_list = []
	scores = {}
	for v in var:
		scores[v] = {}
	for epoch in epochs:
		base_dir = b_dir+'run_67/feature_analysis_epoch'+str(epoch)+'/decoding/'
		these_scores = pkl.load(open(base_dir+'test_scores.pkl','r'))
		for v in var:
			for o in output_nums:
				for t in timesteps:
					row = [epoch, v, o, t, these_scores[('RNN',o,t,v)]]
					out_list.append(row)

	df = pd.DataFrame(out_list, columns=['epoch','variable','output_num','timestep','score'])
	df.to_excel('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/Decoding_by_epoch_run67.xlsx', sheet_name='decoding')


def aggregate_decoding_results2(static=False):

	run_dict = get_run_map()
	to_decode = ['pan_angular_speeds', 'pan_initial_angles']
	for r in range(50):
		to_decode.append('pca_'+str(r))
	#var = ['pan_angular_speeds', 'pan_initial_angles', 'pca_1', 'pca_2', 'pca_3', 'pca_4','pca_5','pca_6','pca_7','pca_8','pca_9', 'pca_10', 'pca_11','pca_12', 'pca_13', 'pca_14','pca_15']
	var = to_decode

	runs = run_dict.keys()
	runs = np.sort(runs)

	if static:
		s_str = '_static'
	else:
		s_str = ''


	out_list = []
	for r in runs:
		run_dir = get_run_dir(r, run_dict[r]['is_server'], run_dict[r]['is_GAN'])
		f_name = run_dir+'feature_analysis/decoding'+s_str+'/test_scores.pkl'
		if os.path.isfile(f_name):
			scores = pkl.load(open(f_name,'r'))
			row = [r, run_dict[r]['name']]
			for i,v in enumerate(var):
				s = -1
				for tup in scores:
					if tup[-1]==v:
						s = scores[tup]
				row.append(s)
			out_list.append(row)

	cols = ['run_num', 'name']+var
	df = pd.DataFrame(out_list, columns=cols)
	df.to_excel('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/Latent_Decoding_Summary'+s_str+'.xlsx', sheet_name='decoding', index=False)


def get_classification_results_for_folder(base_dir, var, layer='RNN', timesteps=[0,4]):

	scores = {}
	for t in timesteps:
		scores[t] = {}
	folders = os.walk(base_dir).next()[1]
	for f in folders:
		idx0 = f.find('ntrain')
		idx1 = f.rfind('_')
		ntrain = int(f[idx0+6:idx1])
		this_dir = base_dir+f+'/'
		these_scores = pkl.load(open(this_dir+'test_scores.pkl','r'))
		for t in timesteps:
			if ntrain not in scores[t]:
				scores[t][ntrain] = []
			scores[t][ntrain].append(these_scores[(layer,0,t,var)])

	return scores


def analyze_classification_results(version):

	b_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
	b_dir_GAN = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/'
	b_dir_GAN2 = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'

	out_list = []

	gan_runs = [307, 276, 277, 452, 322, 342, 478, 499, 511, 573, 668, 662, 635]
	epoch_runs = [67]
	mse_runs = [65, 98, 110, 120, 133, 134, 135, 139, 136, 137, 138, 142, 143]

	epoch_dict = {65: 300, 98: 300, 110: 300, 120: -1, 133: -1, 134: 300, 135: -1, 136: 300, 137: 300, 138: 300, 139: 300, 452: 300, 322: 50, 342: 130, 478: 300, 142: 107, 143: 100, 499: 30, 511: 70, 573: 60, 668: -1, 662: -1, 635: -1}

	all_runs = list(np.copy(gan_runs))
	all_runs.extend(epoch_runs)
	all_runs.extend(mse_runs)

	all_runs.append('pixels')
	all_runs.append('lbp')
	all_runs.append('hog')

	for r in all_runs:#[65,67,98,307]:
		if isinstance(r, str):
			f_name = '/home/bill/Projects/Predictive_Networks/classification_results/version_'+str(version)+'/'+r+'/scores_summary.pkl'
			scores = pkl.load(open(f_name,'r'))
			out_list = append_to_results(out_list, r, -1, scores)
		elif r in mse_runs:
			f_name = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/scores_summary.pkl'
			#f_name = b_dir+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/scores_summary_fc_decoder_relu.pkl'
			epoch = epoch_dict[r]
			scores = pkl.load(open(f_name,'r'))
			out_list = append_to_results(out_list, r, epoch, scores)
		elif r in gan_runs:
			if r==322 or r==342:
				f_name = b_dir_GAN2+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/scores_summary.pkl'
			else:
				f_name = b_dir_GAN+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/scores_summary.pkl'
			#f_name = b_dir_GAN+'run_'+str(r)+'/feature_analysis/classification/version_'+str(version)+'/scores_summary_fc_decoder_relu.pkl'
			if r in epoch_dict:
				epoch = epoch_dict[r]
			else:
				epoch = -1
			scores = pkl.load(open(f_name,'r'))
			out_list = append_to_results(out_list, r, epoch, scores)
		else:
			for epoch in [0,10,25,50,150]:
				f_name = b_dir+'run_67/feature_analysis_epoch'+str(epoch)+'/classification/version_'+str(version)+'/scores_summary.pkl'
				scores = pkl.load(open(f_name,'r'))
				out_list = append_to_results(out_list, r, epoch, scores)

	df = pd.DataFrame(out_list, columns=['run_num','epoch','timestep','n_train','mean','std','min','max'])

	df.to_excel('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/results/Classification_Summary_'+str(version)+'.xlsx', sheet_name='results', index=False)
	#df.to_excel('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/results/Classification_Summary_'+str(version)+'_fc_decoder_relu.xlsx', sheet_name='results', index=False)


def append_to_results(out_list, run_num, epoch, scores):

	for t in scores:
		ntrain = scores[t].keys()
		ntrain = np.sort(ntrain)
		#print 'For t='+str(t)
		for n in ntrain:
			#print ' '+str(n)+': '+str(np.mean(scores[t][n]))+'+-'+str(np.std(scores[t][n]))
			out_list.append([run_num, epoch, t, n, np.mean(scores[t][n]), np.std(scores[t][n]), np.min(scores[t][n]), np.max(scores[t][n])])

	return out_list



def create_projections(run_num, epoch=None, model_epoch=None):

	timestep = 4
	output_num = 0
	layer = 'RNN'
	var = ['pan_initial_angles', 'pan_angular_speeds', 'pca_1', 'pan_angles']

	params_dir = '/home/bill/Data/FaceGen_Rotations/clipset5/'
	calc_idx = range(3000, 4000)

	if run_num==307:
		g_str = 'GAN_'
	elif run_num in [65,67]:
		g_str = ''

	run_dir = '/home/bill/Projects/Predictive_Networks/facegen_'+g_str+'runs_server/run_'+str(run_num)+'/'
	if epoch is None:
		base_dir = run_dir + 'feature_analysis/'
	else:
		base_dir = run_dir + 'feature_analysis_epoch'+str(epoch)+'/'

	if model_epoch is None:
		model_dir = run_dir + 'feature_analysis_epoch'+str(epoch)+'/'
	else:
		model_dir = run_dir + 'feature_analysis_epoch'+str(model_epoch)+'/'
	f_name = model_dir+'decoding/model_info.pkl'
	model_info = pkl.load(open(f_name, 'r'))

	feat_file = base_dir+'features/RNN_features.hkl'
	features = hkl.load(open(feat_file,'r'))
	X = features[output_num][:,timestep]
	X = X[calc_idx]

	params_dict = pkl.load(open(params_dir+'all_params_all.pkl', 'r'))
	for key in params_dict:
		params_dict[key] = params_dict[key][calc_idx]

	projections = {}
	for v in var:
		tup = (layer, output_num, timestep, v)
		betas,_ = model_info[tup]
		b = np.linalg.norm(betas)
		projections[v] = np.dot(X, betas)/b
		projections[v+'_truth'] = get_param_values(v, params_dict, timestep)

	out_dir = base_dir+'projections/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	if model_epoch is None:
		spio.savemat(out_dir+'projections.mat', projections)
	else:
		spio.savemat(out_dir+'projections_modelepoch'+str(model_epoch)+'.mat', projections)



def run_make_feature_hists():

	run_num = 67
	is_server = True
	epochs_to_plot = [0,5,10,50]
	stats = ['corr'] #['norm', 'sparcity']

	layer = 'pool1'
	layer_output_num = 0
	t_step = 0

	for e in epochs_to_plot:
		for s in stats:
			make_feature_stat_hist(run_num, is_server, e, layer, layer_output_num, t_step, s, decode_var='pca_1')


def make_feature_stat_hist(run_num, is_server, epoch, layer, layer_output_num, t_step, stat_name, decode_var=None):

	run_dir = get_run_dir(run_num, is_server)
	save_dir = run_dir+'decoding_plots/'

	features = load_features(run_num, is_server, epoch, layer)
	features = features[layer_output_num][:,t_step]
	features = features.reshape((features.shape[0], np.prod(features.shape[1:])))

	if stat_name=='norm':
		vals = np.linalg.norm(features, axis=1)
	elif stat_name=='sparcity':
		vals = np.zeros(features.shape[0])
		for i in range(features.shape[0]):
			vals[i] = np.mean(features[i]==0.0)
	elif 'corr' in stat_name:
		y = load_params(decode_var)
		y = y[:features.shape[0]]
		vals = np.zeros(features.shape[1])
		for i in range(features.shape[1]):
			if stat_name=='corr':
				vals[i] = np.corrcoef(features[:,i], y)[0,1]
			elif stat_name=='rank_corr':
				vals[i] = ss.spearmanr(features[:,i], y)[0]
		vals[np.isnan(vals)] = 0


	plt.figure()
	plt.hist(vals, normed=True)
	plt.xlabel(stat_name)
	plt.ylabel('Proportion')
	if decode_var is None:
		plt.title('Distribution of '+stat_name+' for layer='+layer+',outnum='+str(layer_output_num)+',t='+str(t_step)+',epoch='+str(epoch))
		plt.savefig(save_dir+'Hist_'+stat_name+'_'+layer+'_output'+str(layer_output_num)+'_t'+str(t_step)+'_epoch'+str(epoch)+'.tif')
	else:
		plt.title('Distribution of '+stat_name+' with '+decode_var+' for layer='+layer+',outnum='+str(layer_output_num)+',t='+str(t_step)+',epoch='+str(epoch))
		plt.savefig(save_dir+'Hist_'+stat_name+'_'+decode_var+'_'+layer+'_output'+str(layer_output_num)+'_t'+str(t_step)+'_epoch'+str(epoch)+'.tif')


def create_random_weights():

	n_create = 10
	model_name = 'facegen_rotation_prednet_twolayer'
	model_params = {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
	out_dir = '/home/bill/Projects/Predictive_Networks/models/facegen_initializations/'

	for i in range(n_create):

		model = initialize_model(model_name, model_params)
		model.save_weights(out_dir+'weights_'+str(i)+'.hdf5')


def run_make_epoch_plots():

	run_num = 67
	is_server = True
	epochs_to_plot = [0, 5, 10, 25, 50, 100] #range(90)#[0,5,10,20,50]

	tag = 'classification'
	make_decoding_by_epoch_plots(run_num, is_server, epochs_to_plot, tag)


def run_make_epoch_plots_pca():

	run_num = 67
	is_server = True
	epochs_to_plot = [0,5,10,25,50]

	make_decoding_by_epoch_plots_pca(run_num, is_server, epochs_to_plot)


def append_angles_to_params():

	f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
	#f_name = '/home/bill/Data/FaceGen_Rotations/clipset4/all_params_train.pkl'
	params_dict = pkl.load(open(f_name,'r'))
	params_dict['pan_angles'] = np.zeros((len(params_dict['pan_initial_angles']),5))
	params_dict['pan_angles_linear'] = np.zeros((len(params_dict['pan_initial_angles']),5))
	for t in range(5):
		params_dict['pan_angles_linear'][:,t] = params_dict['pan_initial_angles']+t*params_dict['pan_angular_speeds']
		a = params_dict['pan_initial_angles']+t*params_dict['pan_angular_speeds']
		for i in range(len(a)):
			while a[i]>2*np.pi:
				a[i] -= 2*np.pi
			while a[i]<0:
				a[i] += 2*np.pi
		params_dict['pan_angles'][:,t] = a
	pkl.dump(params_dict, open(f_name,'w'))


def make_decoding_by_epoch_plots(run_num, is_server, epochs_to_plot, tag=None):

	run_dir = get_run_dir(run_num, is_server)
	save_dir = run_dir+'decoding_plots/decoding_by_epoch/'
	if tag is not None:
		save_dir += tag+'/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	# layers = ['pool0', 'pool1', 'RNN', 'fc_decoder_relu', 'deconv1_relu', 'RNN']
	# layer_output_nums = [0, 0, 0, 0, 0, 0]
	# timesteps_to_use = [0, 0, 0, -1, -1, 4]
	# decode_vars = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'genders', 'ages', 'pan_angular_speeds', 'pan_initial_angles']

	# layers = ['pool0', 'pool1', 'RNN', 'RNN', 'RNN', 'RNN']
	# layer_output_nums = [0, 0, 0, 0, 1, 1]
	# timesteps_to_use = [0, 0, 0, 4, 0, 4]
	# decode_vars = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pan_angular_speeds', 'pan_initial_angles']
	# save_str = '_c-and-h'

	layers = ['RNN', 'RNN', 'RNN', 'RNN']
	layer_output_nums = [0, 0, 1, 1]
	timesteps_to_use = [0, 4, 0, 4]
	decode_vars = ['face_labels','pan_speed_labels']
	save_str = ''

	l_strs = []
	for i,l in enumerate(layers):
		l_strs.append(l+'_'+str(layer_output_nums[i])+'_t'+str(timesteps_to_use[i]))

	all_scores = {}
	for d in decode_vars:
		all_scores[d] = np.zeros( (len(epochs_to_plot), len(layers)) )

	for e_idx,e in enumerate(epochs_to_plot):
		#this_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/decoding/'
		this_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/classification/decoding/'
		these_scores = pkl.load( open(this_dir+'test_scores.pkl','r'))
		for i in range(len(layers)):
			for feat in decode_vars:
				tup = (layers[i], layer_output_nums[i], timesteps_to_use[i], feat)
				all_scores[feat][e_idx, i] = these_scores[tup]


	for d in decode_vars:
		plt.figure()
		plt.plot(epochs_to_plot,all_scores[d])
		#ppl.legend(plt.gca(), layers, loc='best')
		plt.legend(l_strs, loc=0)
		plt.ylabel('Decoding Performace')
		plt.xlabel('Epoch')
		plt.title('Decoding '+d+' over training')
		plt.savefig(save_dir+'Decoding_by_epoch_'+d+save_str+'.tif')
	plt.close('all')


def get_pca_idx_str(idx):

	# for j,k in enumerate(idx):
	# 	if j==0:
	# 		s = str(k)
	# 	else:
	# 		s += '_'+str(k)
	if len(idx)==1:
		s = str(idx[0])
	else:
		s = str(idx[0])+'-'+str(idx[-1])

	return 'idx_'+s


def make_decoding_by_epoch_plots_pca(run_num, is_server, epochs_to_plot, tag=None):

	run_dir = get_run_dir(run_num, is_server)
	save_dir = run_dir+'decoding_plots/decoding_by_epoch_pca/'
	if tag is not None:
		save_dir += tag+'/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	layer = 'RNN'
	pca_idxs = [range(1), range(5), range(10), range(25), range(100)] #[[0],[1],[2],[3],range(5)]
	decode_vars = ['pca_1', 'pca_2', 'pca_3', 'pca_4', 'pca_5', 'pan_angular_speeds', 'pan_angles']
	for output_num in [0,1]:
		for timestep in [0,4]:
			save_str = layer+'_'+str(output_num)+'_t'+str(timestep)


			all_scores = {}
			for d in decode_vars:
				all_scores[d] = np.zeros( (len(epochs_to_plot), len(pca_idxs)) )

			for e_idx,e in enumerate(epochs_to_plot):
				this_dir = run_dir + 'feature_analysis_epoch'+str(e)+'/pca_decoding/'
				for i,idx in enumerate(pca_idxs):
					d = this_dir + get_pca_idx_str(idx) +'/'
					these_scores = pkl.load( open(d+'test_scores.pkl','r'))
					for feat in decode_vars:
						tup = (layer, output_num, timestep, feat)
						all_scores[feat][e_idx, i] = these_scores[tup]

			i_strs = []
			for idx in pca_idxs:
				i_strs.append(get_pca_idx_str(idx))

			plt.figure()
			for d in decode_vars:
				plt.plot(epochs_to_plot,all_scores[d])
				#ppl.legend(plt.gca(), layers, loc='best')
				plt.legend(i_strs, loc=0)
				plt.ylabel('Decoding Performace')
				plt.xlabel('Epoch')
				plt.title('Decoding '+d+' over training with PCA'+'\n'+layer+' output:'+str(output_num)+' t='+str(timestep))
				plt.savefig(save_dir+'PCA_decoding_by_epoch_'+d+'_'+save_str+'.tif')
				plt.clf()
			plt.close('all')





def run_tsne_plot():

	run_num = 67
	is_server = True
	epochs = range(6) #[0,5,10,50]

	# layer = 'RNN'
	# layer_output_num = 0
	# t_step = 4

	layer = 'RNN'
	layer_output_num = [0]
	t_step = [4]
	#epochs = [range(6)]
	#epochs = [[0,5,25,50,150]]
	epochs = [[0,150]]
	embed_method = 'mds'
	ndim=2

	for l in layer_output_num:
		for t in t_step:
			for e in epochs:
				for b in [True]:
					tag = layer+'_'+str(l)+'_t'+str(t)+'_multepochs_'+str(e)
					make_tsne_plot(run_num, is_server, e, layer, l, t, embed_method, tag=tag,ndim=ndim,n_plot=1000)
					# for ei in e:
					# 	tag = layer+'_'+str(l)+'_t'+str(t)+'_singleepochs'
					# 	make_tsne_plot(run_num, is_server, ei, layer, l, t, embed_method, tag=tag)



def load_features(run_num, is_server, epoch, layer):

	run_dir = get_run_dir(run_num, is_server)
	f_name = run_dir+'feature_analysis_epoch'+str(epoch)+'/features/'+layer+'_features.hkl'
	feats = hkl.load(open(f_name, 'r'))

	return feats


def load_params(feat):

	f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
	params_dict = pkl.load(open(f_name,'r'))

	if 'pca' in feat:
		idx = int(feat[feat.find('_')+1:])
		y = params_dict['pca_basis']
		y = y[:,idx]
	elif feat=='genders_binary':
		y = params_dict['genders']>0
		y = y.astype(int)
	else:
		y = params_dict[feat]

	return y


def make_tsne_plot(run_num, is_server, epochs, layer, layer_output_num, t_step,  embed_method, n_plot=500,tag=None,ndim=2):

	run_dir = get_run_dir(run_num, is_server)
	save_dir = run_dir+'decoding_plots/dimensionality_plots/'
	if tag is not None:
		save_dir += tag+'/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	if not isinstance(epochs, list):
		epochs = [epochs]
	np.random.seed(4000)

	for i,epoch in enumerate(epochs):
		f_name = run_dir+'feature_analysis_epoch'+str(epoch)+'/features/'+layer+'_features.hkl'
		feats = hkl.load(open(f_name, 'r'))
		feats = feats[layer_output_num][:,t_step]
		if i==0:
			idx = np.random.permutation(feats.shape[0])[:n_plot]
		feats = feats[idx].astype(np.float64)
		feats = feats.reshape((n_plot, np.prod(feats.shape[1:])))
		if i==0:
			features = feats
		else:
			features = np.vstack((features, feats))

	params_file = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
	params_dict = pkl.load(open(params_file, 'r'))

	var = {}
	var['speed'] = params_dict['pan_angular_speeds'][idx]
	var['angle0'] = params_dict['pan_initial_angles'][idx]
	var['pca_1'] = params_dict['pca_basis'][idx][:,0]
	var['angle'] = params_dict['pan_angles'][idx,t_step]


	if embed_method=='mds':
		clf = skm.MDS(n_components=ndim)
		X = clf.fit_transform(features)
	elif embed_method=='lle':
		clf = skm.LocallyLinearEmbedding()
		X = clf.fit_transform(features)
	elif embed_method=='ltsa':
		clf = skm.LocallyLinearEmbedding(method=embed_method)
		X = clf.fit_transform(features)
	elif embed_method=='tsne':
		X = tsne(features)

	if True:
		f_name = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/MDS_data_ndim'+str(ndim)+'_v2.mat'
		spio.savemat(f_name, {'X': X, 'speed': var['speed'], 'angle0': var['angle0'], 'angle': var['angle'], 'pca_1': var['pca_1']})

	x_min = np.min(X[:,0])
	x_max = np.max(X[:,0])
	y_min = np.min(X[:,1])
	y_max = np.max(X[:,1])
	y_std = np.std(X[:,1])
	x_std = np.std(X[:,0])
	for i,epoch in enumerate(epochs):
		X_plot = X[i*n_plot:(i+1)*n_plot]
		for v in var:
			# c = np.zeros((n_plot, 3))
			# m = 0.8/(np.max(var[v])-np.min(var[v]))
			# c0 = 0.1 - m*np.min(var[v])
			# for i in range(n_plot):
			# 	c[i,0] = var[v][i]*m+c0
			plt.figure()
			plt.scatter(X_plot[:,0], X_plot[:,1], s=30, c=var[v], cmap='bwr')
			plt.ylim([y_min-0.5*y_std,y_max+0.5*y_std])
			plt.xlim([x_min-0.5*x_std,x_max+0.5*x_std])
			plt.axis('off')

			# if is_mds:
			# 	s = 'MDS'
			# else:
			# 	s = 'tSNE'
			s = embed_method
			plt.title(s+' plot for layer='+layer+'_'+str(layer_output_num)+' at t='+str(t_step)+' epoch='+str(epoch)+'\n'+'Colored by '+str(v))
			if len(epochs)>1:
				s2 = 'multepochs'
			else:
				s2 = 'singleepoch'
			plt.savefig(save_dir+s+'_plot_'+layer+'_output'+str(layer_output_num)+'_t'+str(t_step)+'_'+v+'_epoch'+str(epoch)+'_'+s2+'.tif')
	plt.close('all')


def make_final_mds_plots(ndim=2,plot_rank=False):

	# if ndim==2:
	# 	data = spio.loadmat('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/MDS_data.mat')
	# else:
	data = spio.loadmat('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/MDS_data_ndim'+str(ndim)+'_v2.mat')
	epochs = [0,1,5,125]
	n_plot = 500
	X = data['X']

	x_min = np.min(X[:,0])
	x_max = np.max(X[:,0])
	y_min = np.min(X[:,1])
	y_max = np.max(X[:,1])
	y_std = np.std(X[:,1])
	x_std = np.std(X[:,0])
	if ndim==3:
		z_min = np.min(X[:,2])
		z_max = np.max(X[:,2])
		z_std = np.std(X[:,2])

	p_names = {'angle0': 'Initial Angle', 'speed': 'Speed', 'pca_1': 'PCA 1'}
	#c_map = {'angle0': 'BrBG', 'speed': 'RdBu', 'pca_1': 'PuOr'}
	#c_map = {'angle0': 'bwr', 'speed': 'bwr', 'pca_1': 'bwr'}
	c_map = {'angle0': 'viridis', 'speed': 'viridis', 'pca_1': 'viridis'}

	#plt.figure(figsize=(6*len(epochs),24))
	plt.figure(figsize=(6*len(epochs),24))

	for vi,v in enumerate(['speed','pca_1']):
		for ei,epoch in enumerate(epochs):
			X_plot = X[ei*n_plot:(ei+1)*n_plot]
			if ei==0:
				ax1 = plt.subplot(2,len(epochs),len(epochs)*vi+ei+1, aspect='equal', adjustable='box-forced')
			else:
				plt.subplot(2,len(epochs),len(epochs)*vi+ei+1, aspect='equal', adjustable='box-forced', sharex=ax1, sharey=ax1)
			# if ei==0:
			# 	plt.ylabel(p_names[v])
			if plot_rank:
				c = ss.rankdata(data[v])
			else:
				c = data[v]
			if ndim==2:
				plt.scatter(X_plot[:,0], X_plot[:,1], s=100, c=c, cmap=c_map[v])
				plt.ylim([y_min-0.1*y_std,y_max+0.1*y_std])
				plt.xlim([x_min-0.1*x_std,x_max+0.1*x_std])
			else:
				plt.scatter(X_plot[:,0], X_plot[:,1], zs=X_plot[:,2], s=100, c=c, cmap=c_map[v])
				plt.ylim([y_min-0.1*y_std,y_max+0.1*y_std])
				plt.xlim([x_min-0.1*x_std,x_max+0.1*x_std])
				ax1.set_zlim([z_min-0.1*z_std,z_max+0.1*z_std])
			if vi==0:
				plt.title('Epoch '+str(epoch))

			#plt.axis('equal')
			plt.axis('off')
			#plt.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='off')
	plt.show()
	plt.subplots_adjust(hspace=0.0,wspace=0,right=0.9,left=0,top=0.9,bottom=0)
	if plot_rank:
		pr = '_plotrank'
	else:
		pr = ''
	plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/MDS_plot_diffcolors_ndim'+str(ndim)+pr+'.tif')
	plt.close('all')


def run_RNN_heatmap():

	run_num = 67
	is_server = True
	epochs = [0,5,10,50]

	decode_var = 'pca_5'

	for e in epochs:
		make_RNN_decoding_heatmap(run_num, is_server, e, decode_var)


def make_RNN_decoding_heatmap(run_num, is_server, epoch, decode_var):

	run_dir = get_run_dir(run_num, is_server)
	save_dir = run_dir+'decoding_plots/'
	this_dir = run_dir + 'feature_analysis_epoch'+str(epoch)+'/decoding/'

	t_steps = range(0,5)
	nouts = 5
	all_scores = np.zeros((len(t_steps), nouts))

	these_scores = pkl.load( open(this_dir+'test_scores.pkl','r'))
	for ti,t in enumerate(t_steps):
		for l in range(nouts):
			tup = ('RNN', l, t, decode_var)
			all_scores[ti,l] = these_scores[tup]

	plt.figure()
	ax = plt.gca()
	plt.imshow(all_scores, interpolation='none')
	plt.ylabel('t')
	plt.xlabel('output num')
	plt.yticks(range(len(t_steps)))
	ax.set_yticklabels(t_steps)
	plt.xticks(range(5))
	ax.set_xticklabels(['h','c', 'i', 'f','o'])
	plt.colorbar(cmap='Greys')

	plt.savefig(save_dir+'RNN_decoding_heatmap_'+decode_var+'_epoch'+str(epoch)+'.tif')



def load_feature_params(run_num, base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_feature_runs/'):

	f = open(base_save_dir+'run_'+str(run_num)+'/params.pkl', 'r')
	P = pkl.load(f)
	f.close()

	return P


def run_extract_features(param_overrides=None):

	P = get_feature_params(param_overrides)

	f = open(P['data_file'], 'r')
	X = hkl.load(f)
	f.close()
	X = X[P['calculate_idx']]
	X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))

	if P['is_GAN']:
		if 'rand_max' in P:
			rand_max = P['rand_max']
		else:
			rand_max = 0.1
		rand_input = np.random.uniform(low=0.0, high=rand_max, size=(X_flat.shape[0], 128))
		data = {'random_input': rand_input, 'previous_frames': X_flat[:,:P['input_nt']]}
		model,_ = initialize_GAN_models(P['model_name'], P['D_model_name'], P['model_params'], P['D_model_params'])
	elif P['is_autoencoder']:
		data = {'input_frames': X[:,:P['input_nt']]}
		model = initialize_model(P['model_name'], P['model_params'])
	else:
		data = {'input_frames': X_flat[:,:P['input_nt']]}
		model = initialize_model(P['model_name'], P['model_params'])

	model.load_weights(P['weights_file'])

	for i,feat in enumerate(P['feature_layer_names']):
		print 'Computing features for '+feat
		print '   Compiling...'
		if 'RNN' in feat:
			model.nodes[feat].return_sequences = True
			feature_fxn = create_LSTM_feature_fxn(model, feat)
			model.nodes[feat].return_sequences = False
		else:
			feature_fxn = create_feature_fxn(model, feat)

		print '    Extracting features...'
		if P['is_autoencoder']:
			X_tmp = X[:,P['timesteps_to_use'][i]]
			data = {'input_frames': X_tmp}
			features = extract_all_model_features(feature_fxn, model, data, P['batch_size'], P['feature_is_time_expanded'][i], P['input_nt'], P['timesteps_to_use'][i])
			sh = features[0].shape
			features[0] = features[0].reshape((sh[0],1)+sh[1:])
		else:
			features = extract_all_model_features(feature_fxn, model, data, P['batch_size'], P['feature_is_time_expanded'][i], P['input_nt'], P['timesteps_to_use'][i])
		if i==0:
			if not os.path.exists(P['save_dir']):
				os.mkdir(P['save_dir'])

			f = open(P['save_dir'] + 'params.pkl', 'w')
			pkl.dump(P, f)
			f.close()

		f = open(P['save_dir']+feat+'_features.hkl', 'w')
		hkl.dump(features, f)
		f.close()



def run_fit_features(param_overrides=None, P_feature=None):

	P = get_decoding_params(param_overrides)

	if P_feature is None:
		P_feature = load_feature_params(P['feature_run_num'])

	model_info, test_scores, cv_scores = fit_features2(P, P_feature)

	if len(test_scores):

		if not os.path.exists(P['save_dir']):
			os.mkdir(P['save_dir'])
		f = open(P['save_dir']+'model_info.pkl', 'w')
		pkl.dump(model_info, f)
		f.close()

		f = open(P['save_dir']+'test_scores.pkl', 'w')
		pkl.dump(test_scores, f)
		f.close()

		f = open(P['save_dir']+'cv_scores.pkl', 'w')
		pkl.dump(cv_scores, f)
		f.close()

		f = open(P['save_dir']+'test_scores.txt', 'w')
		for tup in test_scores:
			f.write(str(tup)+': '+str(test_scores[tup]))
			f.write("\n")
		f.close()

def get_param_values(feat, params_dict, t):

	if 'pca' in feat:
		idx = int(feat[feat.find('_')+1:])
		y = params_dict['pca_basis']
		y = y[:,idx]
	elif feat=='genders_binary':
		y = params_dict['genders']>0
		y = y.astype(int)
	elif feat=='pan_angles':
		y = params_dict[feat][:,t]
	else:
		y = params_dict[feat]

	return y



def fit_features2(P, P_feature):

	f = open(P['params_file'], 'r')
	params_dict = pkl.load(f)
	f.close()

	model_info = {}
	test_scores = {}
	cv_scores = {}



	for l_num, layer in enumerate(P['layers_to_use']):
		f_name = P_feature['save_dir']+layer+'_features.hkl'
		f = open(f_name, 'r')
		features = hkl.load(f)
		f.close()

		for l_idx in P['layer_outputs_to_use'][l_num]:

			for t in P['timesteps_to_use'][l_num]:

				if t != -1:
					X = features[l_idx][:,t]
				else:
					X = features[l_idx]
				X = X.reshape( (X.shape[0], np.prod(X.shape[1:])))

				for fi,feat in enumerate(P['to_decode']):

					if 'cv_file' in P:
						if P['cv_file'] is not None:
							train_idx,val_idx,test_idx = pkl.load(open(P['cv_file'][fi],'r'))
						else:
							train_idx = np.array(range(P['ntrain']))
							val_idx = P['ntrain']+np.array(range(P['nval']))
							test_idx = P['ntrain']+P['nval']+np.array(range(P['ntest']))
					else:
						train_idx = np.array(range(P['ntrain']))
						val_idx = P['ntrain']+np.array(range(P['nval']))
						test_idx = P['ntrain']+P['nval']+np.array(range(P['ntest']))


					X_train = X[train_idx]
					if len(val_idx)>0:
						X_val = X[val_idx]
					X_test = X[test_idx]

					if 'use_pca' in P:
						if P['use_pca']:
							n_comp = np.max(P['pca_idx'])+1
							pca = skd.PCA(n_components=n_comp)
							X_train = pca.fit_transform(X_train)
							X_val = pca.transform(X_val)
							X_test = pca.transform(X_test)
							X_train = X_train[:,P['pca_idx']]
							X_val = X_val[:,P['pca_idx']]
							X_test = X_test[:,P['pca_idx']]


					y = get_param_values(feat, params_dict, t)
					y = y[P_feature['calculate_idx']]

					y_train = y[train_idx]
					if len(val_idx)>0:
						y_val = y[val_idx]
					y_test = y[test_idx]

					# if True:
					# 	features_test = hkl.load(open('/home/bill/Projects/Predictive_Networks/facegen_feature_runs/run_6/RNN_features.hkl','r'))
					# 	if t != -1:
					# 		X_test = features_test[l_idx][:,t]
					# 	else:
					# 		X_test = features_test[l_idx]
					# 	X_test = X_test.reshape( (X_test.shape[0], np.prod(X_test.shape[1:])))
					# 	params_dict_test = pkl.load(open('/home/bill/Data/FaceGen_Rotations/clipset4/all_params_train.pkl','r'))
					# 	y_test = get_param_values(feat, params_dict_test, t)
					# 	y_test = y_test[:1000]

					tup = (layer, l_idx, t, feat)

					if P['model_params']['method'] == 'adaptive':
						these_scores = {}
						p_list = P['model_params']['start_list']

						while len(p_list)>0:

							for C in p_list:
								these_scores[C] = get_decoding_score(P['fit_model'], X_train, X_val, y_train, y_val, C)

							p_list = []
							all_params = np.array(these_scores.keys())
							all_scores = np.array([these_scores[all_params[k]] for k in range(len(all_params))])
							best_param = get_best_param(all_params, all_scores, P['params_selection_method'])

							if best_param==all_params.min():
								if all_params.min()>P['model_params']['min_param']:
									p_list = [float(all_params.min())/10]
							elif best_param==all_params.max():
								if all_params.max()<P['model_params']['max_param']:
									p_list = [float(all_params.max())*10]
					elif P['model_params']['method'] == 'fixed':
						best_param = P['model_params']['value']
						these_scores = {}

					print ' For '+str(tup)+':'
					print '   shape: '+str(X_train.shape)
					print '   best param: '+str(best_param)
					cv_scores[tup] = these_scores
					test_scores[tup], model_info[tup] = get_decoding_score(P['fit_model'], X_train, X_test, y_train, y_test, best_param, True)
					print '   score: '+str(test_scores[tup])

	return model_info, test_scores, cv_scores



def get_decoding_score(model_name, X_train, X_test, y_train, y_test, C, return_params=False):

	if 'svm' in model_name:
		X = np.vstack( (X_train, X_test) )
		y = np.concatenate( (y_train, y_test) )
		train_idx = range(X_train.shape[0])
		test_idx = range(X_train.shape[0], X.shape[0])
		if model_name=='svm':
			res = libsvm_classify(X, y, train_idx, test_idx, is_kernel_mat = False, C = C, kernel_name = 'linear')
		elif model_name=='svmr':
			res = libsvm_classify(X_train, y_train, train_idx, test_idx, is_kernel_mat = False, C = c, kernel_name = 'linear', s=3)
		score = res[0]
		model_info = []
	elif model_name=='ridge':
		clf = sk.linear_model.Ridge(alpha=C)
		clf.fit(X_train, y_train)
		score = clf.score(X_test, y_test)
		#model_info = clf.get_params()
		model_info = (clf.coef_, clf.intercept_)

	if return_params:
		return score, model_info
	else:
		return score



def get_best_param(C_vals, scores, selection_method):

	idx = np.argsort(C_vals)[::-1]
	C_vals = C_vals[idx]
	scores = scores[idx]
	if selection_method=='best':
		best_param = C_vals[np.argmax(scores)]

	return best_param


def get_best_param2D(C_vals, scores, selection_method):

	idx = np.argsort(C_vals)[::-1]
	C_vals = C_vals[idx]
	scores = scores[idx]
	if selection_method=='best':
		best_param = C_vals[np.argmax(scores.mean(axis=-1))]
	elif params_selection_method=='1sd':
		mean_scores = scores.mean(axis=-1)
		best_score = np.max(mean_scores)
		std_score = scores.std(axis=-1)
		good = mean_scores>=(best_score+std_score)
		idx = np.nonzero(good)[0][0]
		pdb.set_trace()
		best_param = C_vals[idx]

	return best_param


def fit_features(P, P_feature):

	#data_dir = P['data_file'][:P['data_file'].rfind('/')]
	#f_name = data_dir+'face_params.pkl'
	f = open(P['params_file'], 'r')
	params_dict = pkl.load(f)
	f.close()

	model_info = {}
	test_scores = {}
	cv_scores = {}

	for l_num, layer in enumerate(P['layers_to_use']):
		f_name = P_feature['save_dir']+layer+'_features.hkl'
		f = open(f_name, 'r')
		features = hkl.load(f)
		f.close()

		for l_idx in P['layer_outputs_to_use'][l_num]:

			for t in P['timesteps_to_use']:

				X = features[l_idx][:,t]
				X = X.reshape( (X.shape[0], np.prod(X.shape[1:])))

				for feat in P['to_decode']:
					if 'pca' in feat:
						idx = int(feat[feat.find('_')+1:])
						y = params_dict['pca_basis'][P_feature['calculate_idx']]
						y = y[:,idx]
					elif feat=='genders_binary':
						y = params_dict['genders']>0
						y = y.astype(int)
					else:
						y = params_dict[feat][P_feature['calculate_idx']]

					X_train, X_test, y_train, y_test = skcv.train_test_split(X, y, test_size=1-P['train_prop'])
					pdb.set_trace()

					if P['fit_model']=='mds':
						if not os.path.exists(P['save_dir']):
							os.mkdir(P['save_dir'])
					else:

						kf = skcv.KFold(X_train.shape[0], n_folds=P['n_splits'])


						tup = (layer, l_idx, t, feat)
						cv_scores[tup] = np.zeros( (len(P['model_params']), P['n_splits']) )
						for c_idx, c in enumerate(P['model_params']):
							for k, (train_idx, test_idx) in enumerate(kf):
								if P['fit_model']=='svm':
									res = libsvm_classify(X_train, y_train, train_idx, test_idx, is_kernel_mat = False, C = c, kernel_name = 'linear')
									cv_scores[tup][c_idx, k] = res[0]
								elif P['fit_model']=='svmr':
									res = libsvm_classify(X_train, y_train, train_idx, test_idx, is_kernel_mat = False, C = c, kernel_name = 'linear', s=3)
									cv_scores[tup][c_idx, k] = res[0]
								else:
									if P['fit_model']=='ridge':
										clf = sk.linear_model.Ridge(alpha=c)
									clf.fit(X_train[train_idx], y_train[train_idx])
									cv_scores[tup][c_idx, k] = clf.score(X_train[test_idx], y_train[test_idx])
						if P['params_selection_method']=='best':
							best_param = P['model_params'][np.argmax(cv_scores[tup].mean(axis=-1))]
						elif P['params_selection_method']=='1sd':
							mean_scores = cv_scores[tup].mean(axis=-1)
							best_score = np.max(mean_scores)
							std_score = cv_scores[tup].mean(axis=-1)
							good = mean_scores>=(best_score+std_score)
							idx = np.nonzero(good)[0][0]
							pdb.set_trace()
							best_param = P['model_params'][idx]
						clf = sklm.Ridge(alpha=best_param)
						pdb.set_trace()
						clf.fit(X_train, y_train)
						model_info[tup] = clf.get_params()
						test_scores[tup] = clf.score(X_test, y_test)
						print str(tup)+': '+str(test_scores[tup])

	return model_info, test_scores, cv_scores




def create_LSTM_feature_fxn(model, layer_name):

	ins = [model.inputs[name].input for name in model.input_order]
	output = model.nodes[layer_name]
	all_out = output.get_all_outputs(False)
	ys_test = []
	for out in all_out:
		ys_test.append(out)
	pred_fxn = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True, on_unused_input='warn')

	return pred_fxn

def create_LSTM_feature_fxn_only_hidden(model, layer_name):

	ins = [model.inputs[name].input for name in model.input_order]
	output = model.nodes[layer_name]
	all_out = output.get_output(False)
	ys_test = [all_out]
	pred_fxn = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True, on_unused_input='warn')

	return pred_fxn

# t_to_use is a scalar
def extract_all_model_features(feature_fxn, model, X, batch_size, is_time_expanded=False, nt=None, t_to_use=None):

	ins = [X[name] for name in model.input_order]
	nb_sample = len(ins[0])
	outs = []
	batches = make_batches(nb_sample, batch_size)
	index_array = np.arange(nb_sample)
	for batch_index, (batch_start, batch_end) in enumerate(batches):
		print '  batch '+str(batch_index)+'/'+str(len(batches))
		batch_ids = index_array[batch_start:batch_end]
		ins_batch = slice_X(ins, batch_ids)
		batch_outs = feature_fxn(*ins_batch)
		if batch_index==0:
			outs = []
			for data in batch_outs:
				if is_time_expanded:
					if t_to_use is None:
						s = (nb_sample, nt)+data.shape[1:]
					else:
						s = (nb_sample, 1)+data.shape[1:]
				else:
					s = (nb_sample,)+data.shape[1:]
				outs.append(np.zeros(s).astype(np.float32))
		for out_idx, data in enumerate(batch_outs):
			if is_time_expanded:
				s = data.shape
				data = data.reshape((s[0]/nt, nt)+s[1:])
				if t_to_use is not None:
					data = data[:,t_to_use]
					ds = data.shape
					data = data.reshape( (ds[0],1) + ds[1:])
			outs[out_idx][batch_start:batch_end] = data

	return outs




if __name__=='__main__':
	try:
		#run_extract_features()
		#run_fit_features()
		#run_full_decoding_analysis()
		#run_run_full_decoding()
		#run_make_epoch_plots()
		#run_tsne_plot()
		#run_RNN_heatmap()
		#run_make_feature_hists()
		#append_angles_to_params()
		#run_run_pca_decoding()
		#run_make_epoch_plots_pca()
		#create_classification_cv_file(2)
		#run_run_classification_decoding(635,2)
		#aggregate_classification_results(2)
		#analyze_classification_results(2)
		#run_random_classification_decoding()
		#analyze_random_classification()
		#create_random_weights()
		#make_final_mds_plots(2,True)
		#aggregate_decoding_results()
		# model_epoch = None
		# for epoch in [0,5,25,50,150]:
		# 	print epoch
		# 	create_projections(67, epoch, model_epoch)
		#compute_classification_MDS()
		#
		# for r in [662,635]:
		# 	run_run_full_decoding2(r, static=False)
		for r in [144, 65, 635]:
			run_run_full_decoding2(r, static=False, extract_features=False)
		aggregate_decoding_results2(False)

		#aggregate_classification_results2()
		#analyze_classification_results(2)

		#aggregate_decoding_results2()
		#run_feature_classification('hog')

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
