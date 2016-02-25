import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model, load_vgg16_model, load_vgg16_test_model, plot_model, initialize_GAN_models, load_vgg16_test_model2
from prednet import load_mnist, extract_features_graph
from prednet_GAN import translate_im

from copy import deepcopy
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import hickle as hkl
import scipy as sp
import scipy.io as spio
import pickle as pkl
from scipy.misc import imread, imresize
import pandas as pd

from prednet import get_block_num_timesteps

def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *

cname = get_computer_name()

sys.path.append(get_scripts_dir() +'General_Scripts/')

import general_python_functions as gp
sys.path.append('/home/bill/Libraries/keras/')
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import standardize_X, slice_X
from keras.optimizers import *
sys.path.append('/home/bill/Libraries/keras/caffe/')
#from keras.caffe import convert
	#pdb.set_trace()
	#import convert

#sys.path.append('/home/bill/Dropbox/Cox_Lab/caffe/scripts/')
#from caffe_features import create_caffe_features, create_caffe_features2



def get_prednet_params(param_overrides=None):


	P = {}
	P['version'] = 'facegen'
	P['is_GAN'] = True
	P['use_noisy_input'] = False

	if P['version']=='bouncing_ball':
		base_save_dir = '/home/bill/Projects/Predictive_Networks/physics_runs/'
		P['model_name'] = 'bouncing_ball_rnn_twolayer_multsteps'
		P['nt_predict'] = 5
		P['nt_val'] = 20
		P['frames_in_version'] = 'random'
		P['frames_in_params'] = [15, 25]
		P['obj_weights'] = {'output_0': 0.01, 'output_1': 0.01, 'output_2': 0.01, 'output_3': 0.01, 'output_4': 1.0}
		P['model_params'] = {'frame_size': 32, 'batch_size': 16, 'nt_predict': 5}
		P['initial_weights'] = None #('bouncing_ball_autoencoder', {}, ['conv0', 'conv1', 'conv2', 'deconv0', 'deconv1', 'deconv2'], '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/physics_runs/run_7/model_weights.hdf5')
		P['training_files'] = ['/home/bill/Data/Bouncing_Balls/clip_set3/clips.hkl']
		P['evaluation_file'] = '/home/bill/Data/Bouncing_Balls/clip_set3/clips.hkl'
		P['epochs_per_block'] = 500
		P['optimizer'] = 'rmsprop'
		P['n_evaluate'] = P['model_params']['batch_size']

	elif P['version']=='bouncing_ball2':

		if P['is_GAN']:
			base_save_dir = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/'
			P['G_model_name'] = 'bouncing_ball_rnn_twolayer_30x30_G_to_D'
			P['D_model_name'] = 'bouncing_ball_rnn_twolayer_30x30_D'
			P['G_loss'] = {'output': 'GAN_generator_loss2', 'pixel_output': 'mse'}
			P['D_loss'] = {'output': 'GAN_discriminator_loss'}
			P['G_obj_weights'] = {'output': 1.0, 'pixel_output': 0.0}
			P['D_obj_weights'] = {'output': 1.0}
			P['batch_size'] = 8
			P['G_model_params'] = {'use_pixel_output': True, 'batch_size': P['batch_size'], 'use_rand_input': False, 'use_batch_norm': False, 'rand_size': 128, 'encoder_fixed': False, 'decoder_fixed': False, 'n_LSTM': 512,  'pixel_output_flattened': False}
			P['D_model_params'] = {'batch_size': 2*P['batch_size'], 'encoder_fixed': False, 'share_RNN': False, 'use_batch_norm': False, 'n_LSTM': 256}
			P['G_optimizer'] = 'sgd'
			P['D_optimizer'] = 'sgd'
			P['G_learning_rate'] = 0.002
			P['D_learning_rate'] = 0.01
			P['G_momentum'] = 0.4
			P['D_momentum'] = 0.4
			P['G_initial_weights'] = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/run_195/G_model_best_weights.hdf5'
			P['D_initial_weights'] = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/run_195/D_model_best_weights.hdf5'
			P['D_same_conditional'] = True  # will keep conditional data the same for both real and fake examples
			P['G_same_conditional'] = True  # will keep conditional data the same as last D batch
			P['num_batches'] = 75 * 4000/P['batch_size']
			P['plot_frequency'] = 10*4000/P['batch_size']
			P['print_interval'] = 200 #1000/P['batch_size']
			P['val_frequency'] = 200
			P['nb_D_steps'] = 1
			P['nb_G_steps'] = 1
			P['n_eval_int_plot'] = 2
			P['n_evaluate'] = P['batch_size']
			P['frames_in_version'] = 'fixed'
			P['frames_in_params'] = 10
			P['change_nt_frequency'] =  4000/P['batch_size']
			P['kill_criteria'] = [(5 * 4000/P['batch_size'],100), (10* 4000/P['batch_size'],50), (30* 4000/P['batch_size'],30)]
			P['nb_D_pre_epochs'] = 0
			P['model_save_frequency'] = None
			P['is_autoencoder'] = False
		else:
			base_save_dir = '/home/bill/Projects/Predictive_Networks/ball_runs/'
			P['model_name'] = 'bouncing_ball_rnn_twolayer_multsteps_30x30'
			P['frames_in_version'] = 'random'
			P['frames_in_params'] = [5, 15]
			P['obj_weights'] = {'output_0': 1.0}
			P['model_params'] = {'batch_size': 8, 'nt_predict': 1, 'encoder_dropout': False, 'LSTM_dropout': False, 'nfilt': 32}
			P['initial_weights'] = None #('bouncing_ball_autoencoder', {}, ['conv0', 'conv1', 'conv2', 'deconv0', 'deconv1', 'deconv2'], '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/physics_runs/run_7/model_weights.hdf5')
			P['epochs_per_block'] = 5000
			P['optimizer'] = 'rmsprop'
			P['learning_rate'] = 0.001
			P['momentum'] = 0.9
			P['n_evaluate'] = P['model_params']['batch_size']
			P['is_autoencoder'] = False

		P['nt_val'] = 10
		P['nt_predict'] = 5
		P['n_plot'] = P['n_evaluate']
		P['save_model_epochs'] = []
		if P['use_noisy_input']:
			P['noise_file'] = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_training_set_noisy.hkl'
		P['training_files'] = ['/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_training_set.hkl']
		P['evaluation_file'] = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_validation_set.hkl'

	elif P['version']=='facegen':

		if P['is_GAN']:
			base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'
			P['G_model_name'] = 'facegen_rotation_prednet_twolayer_G_to_D'
			P['D_model_name'] = 'facegen_rotation_prednet_twolayer_D'
			P['G_loss'] = {'output': 'GAN_generator_loss2', 'pixel_output': 'mse'}
			P['D_loss'] = {'output': 'GAN_discriminator_loss'}
			P['loss_weights_file_train'] = ['/home/bill/Data/FaceGen_Rotations/clipset4/weightstrain_n3_p3_minw0.1.hkl']
			P['loss_weights_file_val'] = '/home/bill/Data/FaceGen_Rotations/clipset4/weightsval_n3_p3_minw0.1.hkl'
			P['G_obj_weights'] = {'output': 0.0002, 'pixel_output': 1.0}
			P['D_obj_weights'] = {'output': 1.0}
			P['G_model_params'] = {'n_timesteps': 5, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True}
			P['D_model_params'] = {'n_timesteps': 5, 'share_encoder': False, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False, 'use_fc_precat': True, 'RNN_mult': 1.0, 'fc_precat_size': 1024, 'fusion_type': 'early', 'n_LSTM': 1024}
			P['rand_max'] = 0.5
			P['G_optimizer'] = 'rmsprop'
			P['D_optimizer'] = 'sgd'
			P['G_learning_rate'] = 0.001
			P['D_learning_rate'] = 0.01
			P['G_momentum'] = 0.5
			P['D_momentum'] = 0.5
			if cname=='lab_computer2':
				P['batch_size'] = 4  ##used to be 6 **********
			else:
				P['batch_size'] = 6
			P['G_initial_weights'] = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_662/G_model_best_weights.hdf5'
			#('facegen_rotation_prednet_twolayer', {'n_timesteps': 5, 'batch_size': P['batch_size'], 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}, ['conv0','conv1','RNN','fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_runs/run_65/model_best_weights.hdf5', True)
			#'/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_499/G_model_best_weights.hdf5'
			#'/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_463/G_model_best_weights.hdf5' # # #('facegen_rotation_prednet_twolayer', {'n_timesteps': 5, 'batch_size': P['batch_size'], 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}, ['conv0','conv1','RNN','fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_runs/run_77/model_best_weights.hdf5', True) #'/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_262/G_model_best_weights.hdf5'  #('facegen_rotation_prednet_twolayer', {'n_timesteps': 5, 'batch_size': P['batch_size'], 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}, ['conv0','conv1','RNN','fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_runs/run_77/model_best_weights.hdf5', True) #
			#'/home/bill/Projects/Predictive_Networks/models/facegen_GAN_comp342_G_model_best_weights.hdf5'
			P['D_initial_weights'] = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_662/D_model_best_weights.hdf5' #'/home/bill/Projects/Predictive_Networks/models/facegen_GAN_comp342_D_model_best_weights.hdf5' # # #
			P['D_same_conditional'] = True  # will keep conditional data the same for both real and fake examples
			P['G_same_conditional'] = True  # will keep conditional data the same as last D batch
			P['num_batches'] = 10 * 4000/P['batch_size']
			P['plot_frequency'] = 250 #5 * 4000/P['batch_size']
			P['print_interval'] = 250 #200/P['batch_size']
			P['val_frequency'] = 250
			P['model_save_frequency'] = 250 #20*4000/P['batch_size']
			P['nb_D_steps'] = 1
			P['nb_G_steps'] = 1
			P['nb_D_pre_epochs'] = 0 #10*4000/P['batch_size']
			P['nb_G_pre_epochs'] = 0
			P['n_eval_int_plot'] = 6
			P['is_autoencoder'] = False
			P['change_nt_frequency'] =  1*4000/P['batch_size']
			P['kill_criteria'] = [(1 * 4000/P['batch_size'],0.11)] #[(5 * 4000/P['batch_size'],0.11), (10 * 4000/P['batch_size'],0.01), (25 * 4000/P['batch_size'],0.005), (35 * 4000/P['batch_size'],0.004)] #[(0,0.002)] #[(5 * 4000/P['batch_size'],0.01), (10* 4000/P['batch_size'],0.006), (20* 4000/P['batch_size'],0.005)]
		else:
			P['is_autoencoder'] = False

			base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs/'
			if P['is_autoencoder']:
				P['is_denoising'] = True
				P['model_name'] = 'facegen_rotation_autoencoder' #'facegen_rotation_prednet_twolayer'
				P['model_params'] = {'n_FC': 4096, 'batch_size': 4, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
			else:
				P['model_name'] = 'facegen_rotation_prednet_twolayer' #'facegen_rotation_prednet_twolayer'
				P['model_params'] = {'n_timesteps': 6, 'batch_size': 4, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
			P['loss'] = 'mse'
			P['loss_weights_file_train'] = None
			P['loss_weights_file_val'] = None
			P['obj_weights'] = {'output': 1.0}

			P['optimizer'] =  'rmsprop'
			P['learning_rate'] = 0.001

			P['momentum'] = 0.5
			P['initial_weights'] = '/home/bill/Projects/Predictive_Networks/facegen_runs/run_139/model_weights_epoch15.hdf5'  #'/home/bill/Projects/Predictive_Networks/models/facegen_clipset2_run25_model_weights.hdf5'
			P['epochs_per_block'] = 285
			P['save_model_epochs'] = range(0,300,100)

		P['use_reconstruction_loss'] = False
		P['nt_predict'] = 1
		if 'is_autoencoder' in P:
			if P['is_autoencoder']:
				if P['is_denoising']:
					P['nt_val'] = 1
				else:
					P['nt_val'] = 5
			else:
				P['nt_val'] = 5
		else:
			P['nt_val'] = 5
		P['frames_in_version'] = 'fixed'
		P['frames_in_params'] = 5

		P['training_files'] = ['/home/bill/Data/FaceGen_Rotations/clipset4/clipstrain.hkl']
		P['evaluation_file'] = '/home/bill/Data/FaceGen_Rotations/clipset4/clipstest.hkl'

		P['n_evaluate'] = 20
		P['n_plot'] = 20

	P['tag'] = ''
	P['save_model'] = True
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P

def extract_feats_dirty(model, layers, X):

	for i,l_name in enumerate(layers):
		n = 'out'+str(i)
		model.add_output(name=n, input=l_name)

	ins = [model.inputs[name].input for name in model.input_order]
	ys_test = []
	for output_name in model.output_order:
		output = model.outputs[output_name]
		y_test = output.get_output(False)
		ys_test.append(y_test)
	model._predict = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True)
	X_in = [X[name] for name in model.input_order]
	outs = model._predict(*X_in)
	outs_dict = {}
	for n, output_name in enumerate(model.output_order):
		outs_dict[output_name] = outs[n]
	outputs = {}
	for i,l_name in enumerate(layers):
		outputs[l_name] = outs_dict['out'+str(i)]

	return outputs


def get_autoencoder_params(param_overrides=None):

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs/'

	P = {}
	P['model_name'] = 'facegen_rotation_autoencoder'
	P['model_params'] = {}
	P['initial_weights'] = None
	P['corruption'] = {'type': 'none'} #{'type': 'salt_and_pepper', 'count': int(np.round(64*64*0.1))}
	P['training_files'] = ['/home/bill/Data/FaceGen_Rotations/clipset1/clips.hkl']
	P['evaluation_file'] = '/home/bill/Data/FaceGen_Rotations/clipset1/clips.hkl'
	P['batch_size'] = 16
	P['n_train'] = 500
	P['epochs_per_block'] = 20
	P['optimizer'] = 'rmsprop'
	P['n_evaluate'] = 10
	P['n_plot'] = 10
	P['save_model'] = True
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P


def load_params(run_num, base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'):

	f = open(base_save_dir+'run_'+str(run_num)+'/params.pkl', 'r')
	P = pkl.load(f)
	f.close()

	return P

def load_log(run_num, base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'):

	f = open(base_save_dir+'run_'+str(run_num)+'/log.pkl', 'r')
	log = pkl.load(f)
	f.close()

	return log


def load_model_from_run(run_num, base_save_dir, best):

	is_server = 'server' in base_save_dir

	P = load_params(run_num, base_save_dir)
	if is_server:
		idx = P['save_dir'].rfind('/run')
		P['save_dir'] = P['save_dir'][:idx]+'_server'+P['save_dir'][idx:]
	model = initialize_model(P['model_name'], P['model_params'])
	if best:
		model.load_weights(P['save_dir']+'model_best_weights.hdf5')
	else:
		model.load_weights(P['save_dir']+'model_weights.hdf5')

	if 'multsteps' in P['model_name']:
		loss = {}
		for t in range(P['model_params']['nt_predict']):
			loss['output_'+str(t)] = 'mse'
	else:
		loss={'output': 'mse'}
	model.compile(optimizer=P['optimizer'], loss=loss)

	return model


def load_GAN_model_from_run(run_num, base_save_dir, model_str):

	P = load_params(run_num, base_save_dir)

	G_model, D_model = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])

	G_model.load_weights(base_save_dir+'run_'+str(run_num)+'/'+model_str+'.hdf5')

	return G_model



def run_prednet(param_overrides=None):

	P = get_prednet_params(param_overrides)

	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	print 'Save dir '+str(P['save_dir'])
	print 'TRAINING MODEL'
	if P['is_GAN']:
		G_model, D_model, log, G_best_weights, D_best_weights = train_prednet_GAN(P)
	else:
		model, log, best_weights = train_prednet(P)

	f = open(P['save_dir'] + 'log.pkl', 'w')
	pickle.dump(log, f)
	f.close()

	if P['save_model']:
		if P['is_GAN']:
			G_model.save_weights(P['save_dir']+'G_model_weights.hdf5')
			D_model.save_weights(P['save_dir']+'D_model_weights.hdf5')
			if G_best_weights is not None:
				G_model.set_weights(G_best_weights)
				D_model.set_weights(D_best_weights)
				# G_model_best, D_model_best = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])
				# G_model_best.set_weights(G_best_weights)
				# D_model_best.set_weights(D_best_weights)
				G_model.save_weights(P['save_dir']+'G_model_best_weights.hdf5',overwrite=True)
				D_model.save_weights(P['save_dir']+'D_model_best_weights.hdf5',overwrite=True)
		else:
			model.save_weights(P['save_dir']+'model_weights.hdf5')
			if best_weights is not None:
				model.set_weights(best_weights)
				model.save_weights(P['save_dir']+'model_best_weights.hdf5',overwrite=True)

	plot_error_log(P, log)

	print 'EVALUATING MODEL'
	if P['is_GAN']:
		model = G_model
	for v in [True, False]:
		predictions, actual_sequences, pre_sequences = evaluate_prednet(P, model, v)

		if v:
			f = open(P['save_dir'] + 'predictions.pkl', 'w')
			pickle.dump([predictions, actual_sequences], f)
			f.close()

		print 'MAKING PLOTS'
		if P['version']=='facegen':
			make_evaluation_plots_facegen(P, predictions, actual_sequences, pre_sequences, v)
		else:
			make_evaluation_plots(P, predictions, actual_sequences, pre_sequences, v)

	plt.close("all")


def finish_run(run_num):

	sample_num = 1

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'
	P = load_params(run_num, base_save_dir)
	model = load_GAN_model_from_run(run_num, base_save_dir, model_str='G_model_best_weights')
	model = append_predict(model)

	print 'EVALUATING MODEL'
	for v in [True, False]:
		predictions, actual_sequences, pre_sequences = evaluate_prednet(P, model, v)

		if v:
			if sample_num==0:
				f = open(P['save_dir'] + 'predictions.pkl', 'w')
				pickle.dump([predictions, actual_sequences], f)
				f.close()

		print 'MAKING PLOTS'
		if P['version']=='facegen':
			make_evaluation_plots_facegen(P, predictions, actual_sequences, pre_sequences, v, sample_num)
		else:
			make_evaluation_plots(P, predictions, actual_sequences, pre_sequences, v)
	plt.close('all')



def make_plots_for_last_model(run_num):

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
	P = load_params(run_num, base_save_dir)
	model = load_model_from_run(run_num, base_save_dir, False)
	if 'server' in base_save_dir:
		idx = P['save_dir'].rfind('/run')
		P['save_dir'] = P['save_dir'][:idx]+'_server'+P['save_dir'][idx:]+'last_model_'

	for v in [True, False]:
		predictions, actual_sequences, pre_sequences = evaluate_prednet(P, model, v)

		if P['version']=='facegen':
			make_evaluation_plots_facegen(P, predictions, actual_sequences, pre_sequences, v)
		else:
			make_evaluation_plots(P, predictions, actual_sequences, pre_sequences, v)

def make_plots_for_model(run_num, model_str = 'G_model_weights'):

	#run_num = 582
	n_eval = 20
	n_reps = 5
	n_mult_plot = 0

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'

	P = load_params(run_num, base_save_dir)
	model = load_GAN_model_from_run(run_num, base_save_dir, model_str)
	model = append_predict(model)
	#model.compile(optimizer='sgd', loss=P['G_loss'], obj_weights=P['G_obj_weights'])
	P['n_evaluate'] = n_eval
	P['n_plot'] = n_eval
	P['save_dir'] += 'full_evaluation/'
	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])

	predictions = {}
	for r in range(n_reps):
		predictions[r], actual_sequences, pre_sequences = evaluate_prednet(P, model, True)

		if r==0:
			if P['version']=='facegen':
				make_evaluation_plots_facegen(P, predictions[r], actual_sequences, pre_sequences, True)
			else:
				make_evaluation_plots(P, predictions[r], actual_sequences, pre_sequences, True)

	out_dir = P['save_dir']+'multiple_sample_plots/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for i in range(n_mult_plot):
		for r in range(n_reps):
			plt.subplot(1, n_reps, r+1)
			plt.imshow(predictions[r][i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
			plt.axis('off')
		plt.savefig(out_dir+'valclip_'+str(i)+'.tif')


def run_make_plots():

	run_num = 297
	#batches = np.arange(2500, 13500, step=500)
	batches = [5000,13000]
	use_best = True
	save_dir = '/home/bill/Projects/Predictive_Networks/results/facegen_GAN_runs_summary_plots_reduced/'
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	plot_idx = {}
	#plot_idx[0] = [31, 36, 44, 45, 47]
	#plot_idx[1] = [12, 13, 15, 25, 27]
	plot_idx[0] = [49,50,54,60,64]
	plot_idx[1] = [67, 70, 71,72,75]
	plot_idx[2] = [77,78,82,83,85]

	if use_best:
		model_str = 'G_model_best_weights'
		tag = 'run'+str(run_num)+'_bestweights'
		for i in plot_idx:
			make_plots_for_model2(run_num, model_str, plot_idx[i], save_dir, tag+'_plot'+str(i))
	else:
		for batch in batches:
			model_str = 'G_model_weights_batch'+str(batch)

			tag = 'run'+str(run_num)+'_batch'+str(batch)

			for i in plot_idx:
				make_plots_for_model2(run_num, model_str, plot_idx[i], save_dir, tag+'_plot'+str(i))


def append_predict(model):

	ins = [model.inputs[name].input for name in model.input_order]
	ys_test = []
	for output_name in model.output_order:
		output = model.outputs[output_name]
		y_test = output.get_output(False)
		ys_test.append(y_test)
	model._predict = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True)

	return model


def make_plots_for_model2(run_num, model_str, plot_idx, save_dir, tag):

	n_eval = np.max(plot_idx)+1
	n_reps = 1

	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'

	P = load_params(run_num, base_save_dir)
	model = load_GAN_model_from_run(run_num, base_save_dir, model_str)
	#model.compile(optimizer='sgd', loss=P['G_loss'], obj_weights=P['G_obj_weights'])
	model = append_predict(model)
	P['n_evaluate'] = n_eval
	P['n_plot'] = n_eval
	P['save_dir'] += 'full_evaluation/'
	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])

	predictions = {}
	for r in range(n_reps):
		predictions[r], actual_sequences, pre_sequences = evaluate_prednet(P, model, True)


	#out_dir = P['save_dir']
	out_dir = save_dir
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for i,idx in enumerate(plot_idx):
		plt.subplot(n_reps+1, len(plot_idx), 1+i)
		plt.imshow(actual_sequences[idx,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.axis('off')
		for r in range(n_reps):
			plt.subplot(n_reps+1, len(plot_idx), (r+1)*len(plot_idx)+1+i)
			plt.imshow(predictions[r][idx,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
			plt.axis('off')
	plt.savefig(out_dir+'summaryplot_'+tag+'.tif')


def save_final_predictions():

	predictions = {}
	for is_GAN in [True, False]:

		if is_GAN:
			run_num = 668
			base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/'
			model = load_GAN_model_from_run(run_num, base_save_dir, 'G_model_weights_batch1000')
		else:
			run_num = 65
			base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
			model = get_best_facegen_MSE_model()

		model = append_predict(model)

		P = load_params(run_num, base_save_dir)

		P['n_evaluate'] = 200
		P['evaluation_file'] = '/home/bill/Data/FaceGen_Rotations/clipset4/clipstest.hkl'
		X = hkl.load(open(P['evaluation_file'],'r'))
		if is_GAN:
			s = 'GAN'
		else:
			s = 'MSE'
		predictions[s], _, _ = evaluate_prednet(P, model, True)

	f_name = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/final_results/facegen_predictions_submission2.mat'
	spio.savemat(f_name, {'X': X, 'predictions_GAN': predictions['GAN'], 'predictions_MSE': predictions['MSE']})



def plot_error_log(P, log):

	t_str = 'run_'+str(P['run_num'])
	out_dir = P['save_dir']+'plots/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	plt.figure()
	legend_list =[]
	for j in log:
		plt.plot(log[j])
		legend_list.append(j+' '+str(log[j][-1]))
	plt.xlabel('Epoch')
	plt.ylabel('Error')
	plt.legend(legend_list)
	plt.title(t_str)
	plt.show(block=False)
	plt.savefig(out_dir+'error_plot.jpg')


def train_prednet(P):

	model = initialize_model(P['model_name'], P['model_params'])

	model = initialize_weights(P['initial_weights'], model)

	print 'Compiling'
	if 'multsteps' in P['model_name']:
		loss = {}
		for t in range(P['model_params']['nt_predict']):
			loss['output_'+str(t)] = P['loss']
	else:
		loss={'output': P['loss']}

	if P['optimizer']=='rmsprop':
		opt = RMSprop(P['learning_rate'])
	elif P['optimizer']=='sgd':
		opt = SGD(lr=P['learning_rate'], momentum=P['momentum'])
	elif P['optimizer']=='adam':
		opt = Adam(lr=P['learning_rate'], beta_1=P['momentum'])
	model.compile(optimizer=opt, loss=loss, obj_weights=P['obj_weights'])

	block_files = P['training_files']

	best_error = np.inf
	best_weights = None
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()

		if P['loss']=='weighted_mse':
			f = open(P['loss_weights_file_train'][block], 'r')
			X_weights = hkl.load(f)
			f.close()
			f = open(P['loss_weights_file_val'], 'r')
			X_weights_val = hkl.load(f)
			f.close()

		if P['is_autoencoder']:
			if P['is_denoising']:
				X_input = X[:,1]
			else:
				X_input = X[:,0]
			X = X[:,0]

		if P['version']=='bouncing_ball':
			n_ex = X.shape[0]
			n_batches = n_ex/128
			X = X[:(n_batches-1)*128] # 390 is max

			X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
		elif P['version']=='bouncing_ball2':
			if P['use_noisy_input']:
				f = open(P['noise_file'], 'r')
				X_noisy = hkl.load(f)
				f.close()
				nt_diff = X.shape[1]-X_noisy.shape[1]
				if nt_diff>0:
					X = X[:,nt_diff:]
				X_flat = X_noisy.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
			else:
				X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))


			f = open(P['evaluation_file'], 'r')
			X_val = hkl.load(f)
			if X_val.shape[0] % P['model_params']['batch_size'] !=0:
				print 'VALIDATION SET ISNT MULTIPLE OF BATCH SIZE'
				pdb.set_trace()
			if X.shape[0] % P['model_params']['batch_size'] !=0:
				print 'TRAINING SET ISNT MULTIPLE OF BATCH SIZE'
				pdb.set_trace()
			f.close()
			X_val_flat = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]))
			log = {'val_error': np.zeros(P['epochs_per_block']), 'train_error': np.zeros(P['epochs_per_block'])}
			#pdb.set_trace()
			#X = np.concatenate((X,X_val), axis=0)
			#X_flat = np.concatenate((X_flat,X_val_flat), axis=0)
		else:
			# X_val = X[X.shape[0]-P['n_evaluate']:]
			# X_flat_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			# X = X[:X.shape[0]-P['n_evaluate']]
			# X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			f = open(P['evaluation_file'], 'r')
			X_val = hkl.load(f)
			f.close()
			if P['is_autoencoder']:
				if P['is_denoising']:
					X_val_input = X_val[:,1]
				else:
					X_val_input = X_val[:,0]
				X_val = X_val[:,0]
			else:
				X_flat_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
				X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			log = {'val_error': np.zeros(P['epochs_per_block'])}


		if P['version']=='bouncing_ball2':
			log['val_error_single'] = np.zeros(P['epochs_per_block'])
			data_val = {}
			if P['frames_in_version']=='fixed':
				t_val = P['frames_in_params']
			else:
				t_val = (P['frames_in_params'][0]+P['frames_in_params'][1])/2
			w_sum = 0
			for t in range(2, t_val+1, 1):
				data_val[t] = {'input_frames': X_val_flat[:,:t]}
				if t==t_val:
					data_val[t]['weight'] = 100.-float(t_val)
				else:
					data_val[t]['weight'] = 1.0
				w_sum += data_val[t]['weight']
				data_val[t]['output_0'] = X_val[:,t].reshape((X_val.shape[0], 1, X_val.shape[2], X_val.shape[3]))

			for epoch in range(P['epochs_per_block']):
				nt = get_block_num_timesteps(P)
				start_t = np.random.randint(0, X_flat.shape[1]-nt)
				data = {'input_frames': X_flat[:,start_t:start_t+nt]}
				data['output_0'] = X[:,start_t+nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
				print "Epoch: "+str(epoch)
				model.fit(data, batch_size=P['model_params']['batch_size'], nb_epoch=1, verbose=1)
				val_error = 0
				for t in data_val:
					val_error += data_val[t]['weight']*model.evaluate(data_val[t], batch_size=P['model_params']['batch_size'])
				log['val_error'][epoch] = 30*30*val_error/w_sum
				print 'Val error: '+str(log['val_error'][epoch])
				log['train_error'][epoch] = 30*30*model.evaluate(data, batch_size=P['model_params']['batch_size'])
				log['val_error_single'][epoch] = 30*30*model.evaluate(data_val[t_val], batch_size=P['model_params']['batch_size'])
				if log['val_error'][epoch]<best_error:
					best_error = log['val_error'][epoch]
					best_weights = model.get_weights()
		else:
			if P['frames_in_version']=='fixed':

				if P['is_autoencoder']:
					data = {'input_frames': X_input}
					data_val = {'input_frames': X_val_input}
				else:
					nt = get_block_num_timesteps(P)
					data = {'input_frames': X_flat[:,:nt]} #, 'output': X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))}
					data_val = {'input_frames': X_flat_val[:,:nt]}
				if P['version']=='facegen':
					if P['loss']=='weighted_mse':
						data['output'] = np.concatenate( (X_flat[:,nt], X_weights[:,nt]), axis=-1)
						data_val['output'] =  np.concatenate( (X_flat_val[:,nt], X_weights_val[:,nt]), axis=-1)
					else:
						if P['is_autoencoder']:
							data['output'] = X
							data_val['output'] = X_val
						elif P['use_reconstruction_loss']:
							data['output'] = X[:,nt-1]
							data_val['output'] = X_val[:,nt-1]
						else:
							data['output'] = X[:,nt]
							data_val['output'] = X_val[:,nt]
				else:
					if P['model_name']=='bouncing_ball_rnn_twolayer_multsteps':
						for t in range(P['model_params']['nt_predict']):
							data['output_'+str(t)] = X[:,nt+t].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
					else:
						data['output'] = X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
				for epoch in range(P['epochs_per_block']):
					if epoch in P['save_model_epochs']:
						model.save_weights(P['save_dir']+'model_weights_epoch'+str(epoch)+'.hdf5')
					#model.fit(data, batch_size=P['model_params']['batch_size'], nb_epoch=P['epochs_per_block'], verbose=1)
					print "Epoch: "+str(epoch)
					model.fit(data, batch_size=P['model_params']['batch_size'], nb_epoch=1, verbose=1)
					log['val_error'][epoch] = model.evaluate(data_val, batch_size=P['model_params']['batch_size'])
					#log['train_error'][epoch] = model.evaluate(data, batch_size=P['model_params']['batch_size'])
					#print 'Time for val: '+str(time.time()-t0)
					print 'Val error: '+str(log['val_error'][epoch])
					if log['val_error'][epoch]<best_error:
						best_error = log['val_error'][epoch]
						best_weights = model.get_weights()
						model.save_weights(P['save_dir']+'model_best_weights.hdf5',overwrite=True)

			else:
				for epoch in range(P['epochs_per_block']):
					nt = get_block_num_timesteps(P)
					data = {'input_frames': X_flat[:,:nt]} #, 'output': X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))}
					if P['version']=='facegen':
						# if P['loss']=='weighted_mse':
						#
						# else:
						data['output'] = X[:,nt]
					else:
						if P['model_name']=='bouncing_ball_rnn_twolayer_multsteps':
							for t in range(P['model_params']['nt_predict']):
								data['output_'+str(t)] = X[:,nt+t].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
						else:
							data['output'] = X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
					print 'EPOCH: '+str(epoch)
					model.fit(data, batch_size=P['model_params']['batch_size'], nb_epoch=1, verbose=1)

		#nt = P['nt_train']
		#data = {'input_frames': X_flat[:,:nt], 'output': X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))}
		#model.fit(data, batch_size=P['model_params']['batch_size'], nb_epoch=P['epochs_per_block'], verbose=1)

	return model, log, best_weights



def train_prednet_GAN(P):

	G_model, D_model = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])

	#plot_model(G_model)
	#plot_model(D_model)

	G_model = initialize_weights(P['G_initial_weights'], G_model, True)
	D_model = initialize_weights(P['D_initial_weights'], D_model)

	print 'Compiling...'

	if P['G_optimizer']=='rmsprop':
		G_opt = RMSprop(P['G_learning_rate'])
	elif P['G_optimizer']=='sgd':
		G_opt = SGD(lr=P['G_learning_rate'], momentum=P['G_momentum'])
	elif P['G_optimizer']=='adam':
		G_opt = Adam(lr=P['G_learning_rate'], beta_1=P['G_momentum'])
	if P['D_optimizer']=='rmsprop':
		D_opt = RMSprop(P['D_learning_rate'])
	elif P['D_optimizer']=='sgd':
		D_opt = SGD(lr=P['D_learning_rate'], momentum=P['D_momentum'])
	elif P['D_optimizer']=='adam':
		D_opt = Adam(lr=P['D_learning_rate'], beta_1=P['D_momentum'])
	G_model.compile(optimizer=G_opt, loss=P['G_loss'], obj_weights=P['G_obj_weights'])
	D_model.compile(optimizer=D_opt, loss=P['D_loss'], obj_weights=P['D_obj_weights'])
	print 'Finished compiling'


	plt_dir = P['save_dir'] + 'intermediate_plots/'
	need_to_mkdir = True

	log = {'val_error':[]}

	best_error = np.inf
	best_error_batch = 0
	G_best_weights = None
	D_best_weights = None

	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])
	if not os.path.exists(plt_dir):
		os.mkdir(plt_dir)



	block_files = P['training_files']
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()
		if P['version']=='facegen':
			# X_val = X[X.shape[0]-P['n_evaluate']:]
			# X_val_flat = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]*X_val.shape[4]))
			# X = X[:X.shape[0]-P['n_evaluate']]
			# X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			# rand_input_val = np.random.uniform(low=0.0, high=1.0, size=(P['n_evaluate'], P['G_model_params']['rand_size']))
			# n_ex = X.shape[0]

			f = open(P['evaluation_file'], 'r')
			X_val = hkl.load(f)
			idx = [69,3,110,148,117,141]
			for i in range(200):
				if i not in idx:
					idx.append(i)
			X_val = X_val[idx]
			f.close()
			X_val_flat = X_val.reshape((X_val.shape[0], X_val.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4]))
			if P['G_model_params']['use_rand_input']:
				rand_input_val = np.random.uniform(low=0.0, high=P['rand_max'], size=(X_val.shape[0], P['G_model_params']['rand_size']))
			n_ex = X.shape[0]
		elif P['version']=='bouncing_ball2':
			X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
			X = X.reshape((X.shape[0], X.shape[1], 1, X.shape[2], X.shape[3]))
			f = open(P['evaluation_file'], 'r')
			X_val = hkl.load(f)
			f.close()
			X_val_flat = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]))
			X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1, X_val.shape[2], X_val.shape[3]))
			if P['G_model_params']['use_rand_input']:
				rand_input_val = np.random.uniform(low=0.0, high=P['rand_max'], size=(X_val.shape[0], P['G_model_params']['rand_size']))
			n_ex = X.shape[0]

		if P['G_loss']['pixel_output']=='weighted_mse':
			f = open(P['loss_weights_file_train'][block], 'r')
			X_weights = hkl.load(f)
			f.close()
			f = open(P['loss_weights_file_val'], 'r')
			X_weights_val = hkl.load(f)
			f.close()
			X_flat_with_weights = np.concatenate( (X_flat, X_weights), axis=-1)

		y_D = np.zeros((2*P['batch_size'],2), int)
		y_D[:P['batch_size'],1] = 1
		y_D[P['batch_size']:,0] = 1

		y_G = np.zeros((P['batch_size'],2), int)
		y_G[:,1] = 1

		D_data = {}
		G_data = {}
		D_data['output'] = y_D
		G_data['output'] = y_G



		#nt = P['frames_in_params']
		nt = get_block_num_timesteps(P)

		if P['G_model_params']['use_rand_input']:
			G_data_val = {'previous_frames': X_val_flat[:,:nt], 'random_input': rand_input_val}
		else:
			G_data_val = {'previous_frames': X_val_flat[:,:nt]}

		X_shp = X.shape
		n_val = X_val.shape[0]
		plt.figure()

		# if P['nb_D_pre_epochs']>0:
		# 	y_D_all = np.zeros((2*X_shp[0],2), int)
		# 	y_D_all[:X_shp[0],1] = 1
		# 	y_D_all[X_shp[0]:,0] = 1
		#
		# 	D_data_all = {}
		# 	D_data_all['output'] = y_D_all
		# 	D_data_all['previous_frames'] = np.vstack((X_flat[:, :nt], X_flat[:, :nt]))
		#
		# 	if P['G_model_params']['use_rand_input']:
		# 		rand_input = np.random.uniform(low=0.0, high=P['rand_max'], size=(X_shp[0], P['G_model_params']['rand_size']))
		# 		X_next_generated = G_model.predict({'previous_frames': X_flat[:, :nt], 'random_input': rand_input}, batch_size=P['batch_size'])['pixel_output']
		# 	else:
		# 		X_next_generated = G_model.predict({'previous_frames': X_flat[:, :nt]}, batch_size=P['batch_size'])['pixel_output']
		# 	if P['G_model_params']['pixel_output_flattened']:
		# 		X_next_generated = X_next_generated.reshape( (P['batch_size'],)+X_shp[2:] )
		#
		# 	D_data_all['proposed_frames'] =  np.vstack((X[:, nt], X_next_generated))
		#
		# 	idx0 = np.random.permutation(X_shp[0])[:100]
		# 	idx1 = X_shp[0]+np.random.permutation(X_shp[0])[:100]
		# 	idx = np.concatenate((idx0, idx1))
		# 	D_data_test = {}
		# 	for key in D_data_all:
		# 		D_data_test[key] = D_data_all[key][idx]
		# 	for epoch in range(P['nb_D_pre_epochs']):
		# 		print 'D pre epoch '+str(epoch)
		# 		D_model.fit(D_data_all, batch_size=2*P['batch_size'], nb_epoch=1, verbose=1)
		#
		# 		yhat = D_model.predict(D_data_test, batch_size=2*P['batch_size'])['output']
		# 		print '    D prob when true: '+str(yhat[:100,1].mean())
		# 		print '    D prob when false: '+str(yhat[100:,1].mean())

		for batch in range(P['num_batches']):

			if batch<P['nb_D_pre_epochs']:
				skip_G = True
			else:
				skip_G = False
			if batch<P['nb_G_pre_epochs']:
				skip_D = True
			else:
				skip_D = False

			if batch % P['print_interval']==0:
				verbose=1
			else:
				verbose=0

			if batch % P['change_nt_frequency']==0:
				nt = get_block_num_timesteps(P)
				if P['version']=='bouncing_ball2':
					start_t = np.random.randint(0, X_flat.shape[1]-nt)
				else:
					start_t = 0
				if P['G_model_params']['use_rand_input']:
					G_data_val = {'previous_frames': X_val_flat[:,:nt], 'random_input': rand_input_val}
				else:
					G_data_val = {'previous_frames': X_val_flat[:,:nt]}

				#a sanity check
				# if False:
				# 	D_idx0 = np.random.permutation(n_ex)[:P['batch_size']]
				# 	rand_input = np.random.uniform(low=0.0, high=1.0, size=(P['batch_size'], P['G_model_params']['rand_size']))
				# 	G_data['previous_frames'] = X_flat[D_idx0, :nt]
				# 	G_data['random_input'] = rand_input
				# 	gen_data = G_model.predict(G_data)
				# 	yhat_G = gen_data['output']
				# 	D_data['previous_frames'] = X_flat[D_idx0, :nt]
				# 	D_data['proposed_frames'] =  gen_data['pixel_output']
				# 	yhat_D = D_model.predict(D_data)['output']
				# 	pdb.set_trace()


			if not skip_D:
				if verbose:
					print 'Run '+str(P['run_num'])+' Batch: '+str(batch)
					print '  Discriminator:'
				for d_step in range(P['nb_D_steps']):
					D_idx0 = np.random.permutation(n_ex)[:P['batch_size']]
					if P['D_same_conditional']:
						D_idx1 = D_idx0
					else:
						D_idx1 = np.random.permutation(n_ex)[:P['batch_size']]
					D_data['previous_frames'] = np.vstack((X_flat[D_idx0, start_t:start_t+nt], X_flat[D_idx1, start_t:start_t+nt]))
					if P['G_model_params']['use_rand_input']:
						rand_input = np.random.uniform(low=0.0, high=P['rand_max'], size=(P['batch_size'], P['G_model_params']['rand_size']))
						X_next_generated = G_model.predict({'previous_frames': X_flat[D_idx1, start_t:start_t+nt], 'random_input': rand_input})['pixel_output']
					else:
						X_next_generated = G_model.predict({'previous_frames': X_flat[D_idx1, start_t:start_t+nt]})['pixel_output']
					if P['G_model_params']['pixel_output_flattened']:
						X_next_generated = X_next_generated.reshape( (P['batch_size'],)+X_shp[2:] )

					D_data['proposed_frames'] =  np.vstack((X[D_idx0, start_t+nt], X_next_generated))

					if verbose:
						yhat = D_model.predict(D_data)['output']
						print '   Pre-training:'
						print '    D prob when true: '+str(yhat[:P['batch_size'],1].mean())
						print '    D prob when false: '+str(yhat[P['batch_size']:,1].mean())


					D_model.fit(D_data, batch_size=2*P['batch_size'], nb_epoch=1, verbose=verbose)

					if verbose:
						yhat = D_model.predict(D_data)['output']
						print '   Post-training:'
						print '    D prob when true: '+str(yhat[:P['batch_size'],1].mean())
						print '    D prob when false: '+str(yhat[P['batch_size']:,1].mean())

			if not skip_G:
				if verbose:
					print ''
					print '  Generator:'
				for g_step in range(P['nb_G_steps']):
					if g_step==0 and P['G_same_conditional'] and not skip_D:
						G_idx = D_idx0
					else:
						G_idx = np.random.permutation(n_ex)[:P['batch_size']]
					if P['G_model_params']['use_rand_input']:
						rand_input = np.random.uniform(low=0.0, high=P['rand_max'], size=(P['batch_size'], P['G_model_params']['rand_size']))
						G_data['random_input'] = rand_input
					G_data['previous_frames'] = X_flat[G_idx, start_t:start_t+nt]

					if P['G_model_params']['use_pixel_output']:
						if P['G_model_params']['pixel_output_flattened']:
							G_data['pixel_output'] = X_flat_with_weights[G_idx, start_t+nt]
						else:
							G_data['pixel_output'] = X[G_idx, start_t+nt]

					#D_data['previous_frames'] = np.vstack((X_flat[G_idx, start_t:start_t+nt], X_flat[G_idx, start_t:start_t+nt]))
					#ghat = G_model.predict(G_data)['pixel_output']
					#D_data['proposed_frames'] = np.vstack((ghat, ghat))
					#D_feats = extract_feats_dirty(D_model, ['conv0_D','conv1_D','conv1','RNN', 'fc0_D'], D_data)
					#G_feats = extract_feats_dirty(G_model, ['conv0_D','conv1_D','conv1','RNN_D', 'fc0_D'], G_data)
					#pdb.set_trace()

					if verbose:
						yhat = G_model.predict(G_data)['output']
						G_score = yhat[:,1].mean()
						print '   Pre-training:'
						print '    G score: ' + str(G_score)

					G_model.fit(G_data, batch_size=P['batch_size'], nb_epoch=1, verbose=verbose)

					if verbose:
						yhat = G_model.predict(G_data)['output']
						G_score = yhat[:,1].mean()
						print '   Post-training:'
						print '    G score: ' + str(G_score)
						print ''



			if batch % P['val_frequency'] ==0:
				val_predictions = G_model.predict(G_data_val, batch_size=P['batch_size'])['pixel_output']
				if P['G_model_params']['pixel_output_flattened']:
					mse = ((X_val_flat[:,nt] - val_predictions)**2 ).mean()
				else:
					mse = ((X_val[:,nt] - val_predictions)**2 ).mean()
				if P['version']=='bouncing_ball2':
					mse *= 30*30
				log['val_error'].append(mse)
				print '**pixel mse: '+str(mse)
				if mse < best_error:
					best_error = mse
					best_error_batch = batch
					G_best_weights = G_model.get_weights()
					D_best_weights = D_model.get_weights()
					# if mse < 18:
					# 	pdb.set_trace()
					# 	predictions, actual_sequences, pre_sequences = evaluate_prednet(P, G_model, True)
					# 	pdb.set_trace()
					# if mse<0.01:
					# 	G_model.save_weights(P['save_dir']+'G_model_weights_best.hdf5', overwrite=True)
					# 	D_model.save_weights(P['save_dir']+'D_model_weights_best.hdf5', overwrite=True)
				if P['kill_criteria'] is not None:
					for tup in P['kill_criteria']:
						if batch>=tup[0] and mse>=tup[1]:
							return G_model, D_model, log, G_best_weights, D_best_weights

			if batch % P['plot_frequency'] ==0:
				val_predictions = G_model.predict(G_data_val, batch_size=P['batch_size'])['pixel_output']
				if P['G_model_params']['pixel_output_flattened']:
					val_predictions = val_predictions.reshape((n_val,)+X_shp[2:])

				mse = ((X_val[:,nt] - val_predictions)**2 ).mean()
				if P['version']=='bouncing_ball2':
					mse *= 30*30
				#log['val_error'].append(mse)
				print '**pixel mse: '+str(mse)
				for k in range(P['n_eval_int_plot']):
					plt.imshow(val_predictions[k,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
					plt.title('MSE: '+str(mse))
					plt.savefig(plt_dir + 'valclip_'+str(k)+'_batch'+str(batch)+'.jpg')

			if P['model_save_frequency'] is not None:
				if batch % P['model_save_frequency'] ==0:
					if batch>0:
						G_model.save_weights(P['save_dir']+'G_model_weights_batch'+str(batch)+'.hdf5')
						D_model.save_weights(P['save_dir']+'D_model_weights_batch'+str(batch)+'.hdf5')
						G_model.save_weights(P['save_dir']+'G_model_best_weights_batch'+str(batch)+'.hdf5')
						D_model.save_weights(P['save_dir']+'D_model_best_weights_batch'+str(batch)+'.hdf5')


	return G_model, D_model, log, G_best_weights, D_best_weights


def initialize_weights(params, model, force_shape=False, is_GAN=False):

	if params is not None:
		if isinstance(params, str):
			model.load_weights(params)
		else:
			if is_GAN:
				G_model,D_model = initialize_GAN_models(params[0][0], params[0][1], params[1][0], params[1][1])
				if params[0][2]:
					init_model = G_model
				else:
					init_model = D_model
				init_model.load_weights(params[3])
			else:
				init_model = initialize_model(params[0], params[1])
				init_model.load_weights(params[3])
			for layer in params[2]:
				w = init_model.nodes[layer].get_weights()
				w2 = model.nodes[layer].get_weights()
				if force_shape and w[0].shape[0]<w2[0].shape[0]:
					w2[0][:w[0].shape[0]] = w[0]
					if params[-1]:
						w2[0][w[0].shape[0]:] = 0
					w2[1] = w[1]
					w = w2
				model.nodes[layer].set_weights(w)

	return model


def evaluate_prednet(P, model, is_validation):

	#if P['version']=='bouncing_ball2':
	if is_validation:
		f_name = P['evaluation_file']
	else:
		f_name = P['training_files'][0]
	f = open(f_name, 'r')
	X = hkl.load(f)
	f.close()
	X = X[:P['n_evaluate']]
	if 'is_autoencoder' in P:
		if P['is_autoencoder']:
			if P['is_denoising']:
				X_data = X[:,1]
			else:
				X_data = X[:,0]
	# else:
	# 	f = open(P['evaluation_file'], 'r')
	# 	X = hkl.load(f)
	# 	f.close()
	# 	if is_validation:
	# 		X = X[-P['n_evaluate']:]
	# 	else:
	# 		X = X[:P['n_evaluate']]
	s = X.shape

	predictions = np.zeros((P['n_evaluate'], P['nt_predict']) + s[2:])
	actual_sequences = X[:,P['nt_val']:P['nt_val']+P['nt_predict']]
	if P['version']=='facegen':
		if 'is_autoencoder' in P:
			if P['is_autoencoder']:
				if P['is_denoising']:
					pre_sequences = X
				else:
					pre_sequences = X[:,P['nt_val']-2:P['nt_val']]
			else:
				pre_sequences = X[:,P['nt_val']-2:P['nt_val']]
		else:
			pre_sequences = X[:,P['nt_val']-2:P['nt_val']]
	else:
		pre_sequences = X[:,P['nt_val']-P['nt_predict']:P['nt_val']]
	input_frames = np.zeros((P['n_evaluate'], P['nt_val']+P['nt_predict']-1, np.prod(s[2:])))
	input_frames[:,:P['nt_val']] = X[:,:P['nt_val']].reshape((s[0], P['nt_val'], np.prod(s[2:])))
	for i in range(P['nt_predict']):
		if i>0:
			input_frames[:,P['nt_predict']+i] = predictions[:,i-1].reshape( (s[0], np.prod(s[2:]) ))

		if P['is_GAN']:
			out_name = 'pixel_output'
			data = {'previous_frames': input_frames[:,i:(P['nt_val']+i)]}
			if P['G_model_params']['use_rand_input']:
				data['random_input'] = np.random.uniform(low=0.0, high=P['rand_max'], size=(P['n_evaluate'], P['G_model_params']['rand_size']))
		else:
			if 'multsteps' in P['model_name']:
				out_name = 'output_0'
			else:
				out_name = 'output'
			data = {'input_frames': input_frames[:,i:(P['nt_val']+i)]}
			if 'is_autoencoder' in P:
				if P['is_autoencoder']:
					data = {'input_frames': X_data}
		if 'batch_size' in P:
			b_size = P['batch_size']
		elif 'model_params' in P:
				if 'batch_size' in P['model_params']:
					b_size = P['model_params']['batch_size']
				else:
					b_size = 4
		else:
			b_size = 4
		yhat = model.predict(data, batch_size=b_size)[out_name]
		predictions[:,i] = yhat.reshape((s[0],)+s[2:])

	return predictions, actual_sequences, pre_sequences


def make_evaluation_plots(P, predictions, actual_sequences, pre_sequences, is_validation):

	t_str = 'run_'+str(P['run_num'])
	out_dir = P['save_dir']+'plots/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	nt = predictions.shape[1]

	if is_validation:
		s = 'val'
	else:
		s = 'train'

	plt.figure()
	for i in range(P['n_plot']):
		for t in range(nt):
			plt.subplot(3, nt, t+1)
			plt.imshow(pre_sequences[i,t], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
			plt.gca().axes.get_xaxis().set_ticks([])
			plt.gca().axes.get_yaxis().set_ticks([])
			if t==0:
				plt.ylabel('Prior')
				plt.title(t_str + ' '+s+'clip_'+str(i))

			plt.subplot(3, nt, t+nt+1)
			plt.imshow(actual_sequences[i,t], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
			plt.gca().axes.get_xaxis().set_ticks([])
			plt.gca().axes.get_yaxis().set_ticks([])
			if t==0:
				plt.ylabel('Actual')

			plt.subplot(3, nt, t+2*nt+1)
			plt.imshow(predictions[i,t], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
			plt.gca().axes.get_xaxis().set_ticks([])
			plt.gca().axes.get_yaxis().set_ticks([])
			if t==0:
				plt.ylabel('Predicted')

		plt.savefig(out_dir+s+'clip_'+str(i)+'.jpg')


def make_evaluation_plots_facegen(P, predictions, actual_sequences, pre_sequences, is_validation, sample_num=0):

	t_str = 'run_'+str(P['run_num'])
	if sample_num==0:
		out_dir = P['save_dir']+'plots/'
	else:
		out_dir = P['save_dir']+'plots_sample'+str(sample_num)+'/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	nt = predictions.shape[1]

	if is_validation:
		s = 'val'
	else:
		s = 'train'

	plt.figure()
	for i in range(P['n_plot']):
		plt.subplot(2, 2, 1)
		plt.imshow(pre_sequences[i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.title(t_str + ' '+s+'clip_'+str(i))
		plt.ylabel('Previous-1')

		plt.subplot(2, 2, 2)
		plt.imshow(pre_sequences[i,1,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.ylabel('Previous')

		plt.subplot(2, 2, 3)
		plt.imshow(actual_sequences[i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.ylabel('Actual')

		plt.subplot(2, 2, 4)
		plt.imshow(predictions[i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.ylabel('Predicted')

		plt.savefig(out_dir+s+'clip_'+str(i)+'.jpg')



def run_autoencoder(param_overrides=None):

	P = get_autoencoder_params(param_overrides)

	print 'TRAINING MODEL'
	model= train_autoencoder(P)

	os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	if P['save_model']:
		model.save_weights(P['save_dir']+'model_weights.hdf5')

	print 'EVALUATING MODEL'
	for v in [True, False]:
		reconstructions, original_X, corrupted_X = evaluate_autoencoder(P, model, v)

		#f = open(P['save_dir'] + 'predictions.pkl', 'w')
		#pickle.dump([predictions, actual_sequences], f)
		#f.close()

		print 'MAKING PLOTS'
		make_autoencoder_evaluation_plots(P, reconstructions, original_X, corrupted_X, v)


def train_autoencoder(P):

	model = initialize_model(P['model_name'], P['model_params'])
	#plot_model(model)

	print 'Compiling'
	model.compile(optimizer=P['optimizer'], loss={'output': 'mse'})

	block_files = P['training_files']
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()
		nt = X.shape[1]

		sampled_X = np.zeros((P['n_train'], 1, X.shape[-2], X.shape[-1]))

		t = np.random.randint(nt, size=P['n_train'])
		for i in range(P['n_train']):
			sampled_X[i,0] = X[i, t[i]]
		corrupted_X = np.copy(sampled_X)
		if P['corruption']['type']=='salt_and_pepper':
			for i in range(P['n_train']):
				rows = np.random.randint(X.shape[2], size=P['corruption']['count'])
				cols = np.random.randint(X.shape[2], size=P['corruption']['count'])
				for j in range(P['corruption']['count']):
					if corrupted_X[i,0,rows[j],cols[j]]==0:
						corrupted_X[i,0,rows[j],cols[j]] = 1
					else:
						corrupted_X[i,0,rows[j],cols[j]] = 0

		data = {'input_frames': corrupted_X, 'output': sampled_X}

		model.fit(data, batch_size=P['batch_size'], nb_epoch=P['epochs_per_block'], verbose=1)

	return model




def evaluate_autoencoder(P, model, validation=True):

	f = open(P['evaluation_file'], 'r')
	X = hkl.load(f)
	f.close()
	if validation:
		X = X[-P['n_evaluate']:]
	else:
		X = X[:P['n_evaluate']]
	s = X.shape

	original_X = np.zeros((s[0], 1, s[-1], s[-1]))

	t = np.random.randint(s[1], size=s[0])
	for i in range(s[0]):
		original_X[i,0] = X[i, t[i]]

	corrupted_X = np.copy(original_X)
	if P['corruption']['type']=='salt_and_pepper':
		for i in range(s[0]):
			rows = np.random.randint(s[-1], size=P['corruption']['count'])
			cols = np.random.randint(s[-1], size=P['corruption']['count'])
			for j in range(P['corruption']['count']):
				if corrupted_X[i,0,rows[j],cols[j]]==0:
					corrupted_X[i,0,rows[j],cols[j]] = 1
				else:
					corrupted_X[i,0,rows[j],cols[j]] = 0

	data = {'input_frames': original_X}
	reconstructions = model.predict(data)['output']

	return reconstructions, original_X, corrupted_X


def make_autoencoder_evaluation_plots(P, reconstructions, original_X, corrupted_X, is_validation):

	t_str = 'run_'+str(P['run_num'])
	out_dir = P['save_dir']+'plots/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	if is_validation:
		s = 'val'
	else:
		s = 'train'

	plt.figure()
	for i in range(P['n_plot']):

		plt.subplot(1, 3, 1)
		plt.imshow(original_X[i,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.axis('off')
		plt.xlabel('Original')

		plt.subplot(1, 3, 2)
		plt.imshow(corrupted_X[i,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.axis('off')
		plt.xlabel('Corrupted')

		plt.subplot(1, 3, 3)
		plt.imshow(reconstructions[i,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.axis('off')
		plt.xlabel('Reconstruction')

		plt.title(t_str + ' '+s+'clip_'+str(i))

		plt.savefig(out_dir+s+'clip_'+str(i)+'.jpg')

def test_caffe_model():



	img_file = '/home/bill/Libraries/caffe/examples/images/cat.jpg'

	feats = create_caffe_features(img_file, scores_bool = True)

	caffe_root = '/home/bill/Libraries/caffe/'
	model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
	model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

	#model_def = '/home/bill/Libraries/caffe/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt'
	#model_weights = '/home/bill/Libraries/caffe/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel'
	model = convert.caffe_to_keras(prototext=model_def, caffemodel=model_weights)

	imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
	labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

	image = imread(img_file)
	image = imresize(image, (256, 256)).astype(np.float32)
	image = image[:, :, (2, 1, 0)]
	# mean subtraction (get mean from model file?..hardcoded for now)
	image = image - np.array([103.939, 116.779, 123.68])
	# resize
	#image /= 255
	# get channel in correct dimension
	image = np.transpose(image, (2, 0, 1))
	X = image.reshape((1, 3, 256, 256))


	feat = extract_features_graph(model, ['output_prob'], {'conv1': X})
	pdb.set_trace()

def test_vgg():

	is_test = False

	if is_test:
		model = load_vgg16_test_model2()
	else:
		model = load_vgg16_model()

	img_file = '/home/bill/Libraries/caffe/examples/images/cat.jpg'

	nx = 224
	image = imread(img_file).astype(np.float32)
	if image.shape[0]>image.shape[1]:
		d = int(np.round( float(image.shape[0]-image.shape[1])/2))
		image = image[d:d+image.shape[1],:,:]
	elif image.shape[0]<image.shape[1]:
		d = int(np.round( float(image.shape[1]-image.shape[0])/2))
		image = image[:,d:d+image.shape[0],:]
	image = imresize(image, (nx, nx)).astype(np.float32)
	# just handcropping
	#pdb.set_trace()
	#image = image[66:66+224,66:66+224,:]
	#pdb.set_trace()

	data = spio.loadmat('test_rand_im.mat')
	image = data['im']

	# mean_im = np.zeros((nx, nx, 3))
	# mean_im[:,:,0] = 123.68
	# mean_im[:,:,1] = 116.779
	# mean_im[:,:,2] = 103.939
	# image -= mean_im
	image = np.transpose(image, (2, 0, 1))
	X = image.reshape((1, 3, nx, nx))

	#np.sum(X[0,:,:3,:3]*model.nodes['conv1_1'].get_weights()[0][0])+model.nodes['conv1_1'].get_weights()[1][0]


	if is_test:
		X = np.random.rand(1,3,3,3)
		model.compile('sgd', {'output': 'mse'})
		feat2 = model.predict({'input': X})
		feat_predict0 = feat2['output'][0][0]
		feat_predict1 = feat2['output'][0][-1]
		feat = extract_features_graph(model, ['conv1_1'], {'input': X})
		feat_extract0 = feat['conv1_1'][0][0]
		feat_extract1 = feat['conv1_1'][0][-1]

		w0 = model.nodes['conv1_1'].get_weights()[0][0]
		x0 = X[0]


		x1 = feat['conv1_1'][0][0,0]
		import scipy.signal as sps
		con = sps.convolve(x0, w0, 'valid')
		pdb.set_trace()
		print 'conv:'
		print con[0][0][:10]
		print 'from model:'
		print feat['conv1_1'][0][0][0][:10]
		pdb.set_trace()

	else:
		feat = extract_features_graph(model, ['prob','conv1_1','pool1'], {'input': X})

		c = feat['prob'].argmax()
		caffe_root = '/home/bill/Libraries/caffe/'
		imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
		labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

		print 'Predicted Label: '+labels[c]
		print 'Score: '+str(feat['prob'][0,c])

		print 'from model'
		print feat['conv1_1'][0,0,1:5,1:5]

		# this is messed up
		# w,b = model.nodes['conv1_1'].get_weights()
		# val = np.sum(w[0,:,:,::-1][0,:,::-1]*X[0,:,:3,:3])+b[0]
		# print 'from calc'
		# print val

		#plt.imshow(feat['conv1_1'][0,0])
		#plt.colorbar()
		#plt.show()

		pdb.set_trace()

def test_keras_conv():

	model = initialize_model('conv_test')
	X = np.random.rand(1,3,3,3)
	#X = np.ones((1,3,3,3))
	model.compile('sgd', {'output': 'mse'})
	#model.nodes['conv'].set_weights((np.ones((1,3,3,3)), np.zeros(1)))
	feat = model.predict({'input': X})
	print 'keras:'
	feat = feat['output']
	print feat
	w,b = model.nodes['conv'].get_weights()
	print 'option 1:'
	print np.sum(w*X[0])+b
	print 'option 2:'
	print np.sum(w[::-1,::-1,:]*X[0])+b
	print 'option 3:'
	print np.sum(w[::-1,::-1,::-1]*X[0])+b
	print 'option 4:'
	print np.sum(w[:,::-1,::-1]*X[0])+b
	print 'option 5:'
	print np.sum(w[:,:,:,::-1][:,:,::-1]*X[0])+b

	pdb.set_trace()


def create_mnist_shift_videos3():

	out_dir = '/home/bill/Data/MNIST/MNIST_Shift_Clips/clipset_1/'
	n_train = 50000

	(X_orig, y_orig), _ = load_mnist()
	y_orig = np.argmax(y_orig, axis=-1)
	X = {}
	y = {}
	X['train'] = X_orig[:n_train]
	y['train'] = y_orig[:n_train]
	X['val'] = X_orig[n_train:]
	y['val'] = y_orig[n_train:]
	for t in ['val', 'train']:
		n_ex = X[t].shape[0]
		frame_size = X[t].shape[-1]
		idxs = {}
		for i in range(10):
			idxs[i] = np.nonzero(y[t]==i)[0]

		clips = np.zeros((n_ex, 3, 1, frame_size, frame_size)).astype(np.float32)
		clip_info = []

		for i in range(n_ex):
			print 'Starting clip '+t+' '+str(i)
			j = np.random.randint(3)
			k = np.random.randint(len(idxs[j]))
			idx = idxs[j][k]

			clips[i,0,0] = X[t][idx]
			clips[i,1] = X[t][i]
			clip_dict = {'first_num': j, 'first_num_idx': idx}

			this_class = y[t][i]
			k = np.random.randint(len(idxs[this_class]))
			idx2 = idxs[this_class][k]

			clip_dict['second_num'] = this_class
			clip_dict['second_num_idx'] = i
			clip_dict['third_num'] = this_class
			clip_dict['third_num_idx'] = idx2


			if j==0:
				shift_x = np.random.randint(-2, 3)
				shift_y = np.random.randint(-8, -2)
			elif j==1:
				shift_x = np.random.randint(-2, 3)
				shift_y = np.random.randint(3, 9)
			elif j==2:
				if np.random.rand()<0.5:
					shift_y = np.random.randint(3, 9)
				else:
					shift_y = np.random.randint(-8, -2)
				shift_x = np.random.randint(-2, 3)

			clips[i,2,0] = translate_im(X[t][idx2,0], shift_x, shift_y)
			clip_dict['num_shift_x'] = shift_x
			clip_dict['num_shift_y'] = shift_y
			clip_info.append(clip_dict)

		f = open(out_dir + t+'_clips.hkl', 'w')
		hkl.dump(clips, f)
		f.close()
		f = open(out_dir + t+'_clip_info.pkl', 'w')
		pkl.dump(clip_info, f)
		f.close()


def plot_logs(base_dir, run_nums, tags):

	logs = {}

	for i,r in enumerate(run_nums):
		f = open(base_dir + 'run_'+str(r)+'/log.pkl', 'r')
		logs[tags[i]] = pkl.load(f)
		f.close()

	plt.figure()
	legend_list =[]
	for t in tags:
		for j in ['val_error', 'train_error']:
			plt.plot(logs[t][j])
			legend_list.append(t+' '+j)
	plt.legend(legend_list)
	plt.show(block=False)
	plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/error_plot.jpg')


def save_predictions2mat(run_num):

	d = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_'+str(run_num)+'/'
	fname = d+'predictions.pkl'
	f = open(fname, 'r')
	predictions, actual_sequences = pkl.load(f)
	f.close()

	P = load_params(run_num)
	f = open(P['evaluation_file'], 'r')
	X = hkl.load(f)
	f.close()
	X = X[-P['n_evaluate']:]
	s = X.shape

	if P['version']=='facegen':
		pre_sequences = X[:,P['nt_val']-2:P['nt_val']]
	else:
		pre_sequences = X[:,P['nt_val']-P['nt_predict']:P['nt_val']]

	spio.savemat(d+'predictions.mat',{'predictions': predictions, 'actual_sequences': actual_sequences, 'pre_sequences': pre_sequences})


def print_learning_rates(start_run, end_run):

	f = open('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/results/ball_GAN_runs_summary_'+str(start_run)+'-'+str(end_run)+'.txt', 'w')

	for r in range(start_run, end_run+1):
		try:
			P = load_params(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/')
			f.write('Run '+str(r))
			f.write("\n")
			f.write('  opt: '+str(P['G_optimizer']))
			f.write("\n")
			f.write('  G lr: '+str(P['G_learning_rate']))
			f.write("\n")
			f.write('  D lr: '+str(P['D_learning_rate']))
			f.write("\n")
			f.write('  mom: '+str(P['G_momentum']))
			f.write("\n")
			f.write('  frames in: '+str(P['frames_in_version']))
			f.write("\n")
			log = load_log(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/')
			f.write('  min error: '+str(min(log['val_error'])))
			f.write("\n")
			f.write('  last error: '+str(log['val_error'][-1]))
			f.write("\n")
			f.write("\n")
		except:
			print 'Couldnt load params: '+str(r)

	f.close()

def make_results_spreadsheet(start_run, end_run, exp, is_server):

	out = []
	if is_server:
		s = '_server'
	else:
		s = ''

	for r in range(start_run, end_run+1):
		try:
			P = load_params(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/'+exp+'_runs'+s+'/')
			log = load_log(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/'+exp+'_runs'+s+'/')
			print r, log['val_error'][-1], min(log['val_error'])
			#row = [r, P['training_files'][0], P['G_optimizer'], P['G_model_params']['use_batch_norm'], P['G_learning_rate'], P['D_learning_rate'], P['G_momentum'], P['frames_in_version'], log['val_error'][-1], min(log['val_error'])]
			# if P['G_initial_weights'] is None:
			# 	iw = 0
			# else:
			# 	iw = 1
			row = [r, P['training_files'][0], str(P['G_initial_weights']), str(P['D_initial_weights']), P['G_optimizer'], P['D_model_params']['share_encoder'], P['D_model_params']['encoder_params_fixed'], P['D_model_params']['share_RNN'], P['G_loss']['pixel_output'], P['G_learning_rate'], P['D_learning_rate'], P['G_obj_weights']['output'], P['G_obj_weights']['pixel_output'], P['G_momentum'], P['frames_in_version'], P['batch_size'], P['num_batches'], log['val_error'][-1], min(log['val_error'])]
			out.append(row)
		except:
			print 'Error with '+str(r)

	df = pd.DataFrame(out, columns = ['run', 'train_file', 'initial_weights', 'D_initial_weights', 'optimizer', 'D_share_encoder', 'D_encoder_fixed', 'D_share_RNN', 'G_pixel_loss', 'G_lr', 'D_lr', 'GAN_weight','MSE_weight', 'momentum', 'frames_in_version', 'batch_size', 'n_epochs', 'last_error', 'min_error'])
	#df = pd.DataFrame(out, columns = ['run', 'train_file', 'optimizer', 'use_batch_norm', 'G_lr', 'D_lr', 'momentum', 'frames_in_version', 'last_error', 'min_error'])
	out_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/results/'
	if not os.path.exists(out_dir):
		out_dir = '/home/bill/Projects/Predictive_Networks/results/'
		s = '_server'
	df.to_excel(out_dir+exp+'_summary'+s+'_runs'+str(start_run)+'-'+str(end_run)+'.xlsx', index=False, sheet_name='sheet1')



def print_stuff(start_run, end_run, exp, is_server):

	out = []
	if is_server:
		s = '_server'
	else:
		s = ''

	for r in range(start_run, end_run+1):
		try:
			P = load_params(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/'+exp+'_runs'+s+'/')
			log = load_log(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/'+exp+'_runs'+s+'/')
			print 'Run '+str(r)
			print '   min error: '+str(min(log['val_error']))
			print '   last error: '+str(log['val_error'][-1])
			print '   initial_weights: '+str(P['G_initial_weights'])
			print '   pixel weight: '+str(P['G_obj_weights']['pixel_output'])
		except:
			print 'Error with '+str(r)


def calculate_naive_error_rate():

	file_name = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_testing_set.hkl'
	f = open(file_name, 'r')
	X = hkl.load(f)
	f.close()
	X_hat = X[:,1:]
	X = X[:,:-1]
	d = 30*30*(X_hat-X)**2
	d = d.reshape((d.shape[0], np.prod(d.shape[1:])))
	error_rate = 30*30*np.mean((X_hat-X)**2)
	std_score = np.std(np.mean(d,axis=-1))
	pdb.set_trace()
	print 'error rate:'+str(error_rate)
	print 'std:'+str(std_score)



def check_random_weights(run_num, base_save_dir):

	P = load_params(run_num, base_save_dir)
	G_model, D_model = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])
	G_model.load_weights(base_save_dir+'run_'+str(run_num)+'/'+'G_model_weights.hdf5')

	if 'bouncing_ball' in P['version']:
		W = G_model.nodes['fc0'].get_weights()
		pdb.set_trace()


def produce_prediction_for_dataset():

	P = {}
	base_dir = '/home/bill/Data/Bouncing_Balls/clip_set4/predictions/'

	P['save_dir'] = base_dir + 'run_' + str(gp.get_next_run_num(base_dir)) + '/'


	P['model_base_save_dir'] = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs/'
	P['model_run_num'] = 256
	P['data_file'] = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_validation_set.hkl'
	P['nt_in'] = 10
	P['batch_size'] = 8
	P['start_at_beginning'] = True

	P_run = load_params(P['model_run_num'], P['model_base_save_dir'])
	if P_run['is_GAN']:
		model = load_GAN_model_from_run(P['model_run_num'], P['model_base_save_dir'], 'G_model_best_weights')
		model = append_predict(model)
	else:
		model = load_model_from_run(P['model_run_num'], P['model_base_save_dir'], best=True)

	f = open(P['data_file'], 'r')
	X = hkl.load(f)
	f.close()
	s = X.shape

	X_flat = X.reshape(s[:2]+(np.prod(s[2:]),))

	predictions = np.zeros((s[0],s[1]-1)+s[2:]).astype(np.float32)
	scores = np.zeros(s[1]-1)
	scores_per_clip = np.zeros((s[0], s[1]-1))

	if P['start_at_beginning']:
		t0 = 0
	else:
		t0 = P['nt_in']-1

	for t in range(t0, s[1]-1):
		print 'Computing for t='+str(t)
		t_start = t - P['nt_in']+1
		if t_start<0:
			t_start = 0

		if P_run['is_GAN']:
			out_name = 'pixel_output'
			data = {'previous_frames': X_flat[:,t_start:t+1]}
			if 'use_rand_input' in P_run['G_model_params']:
				use_rand = P_run['G_model_params']['use_rand_input']
			else:
				use_rand = True
			if use_rand:
				if 'rand_max' in P_run:
					rand_max = P_run['rand_max']
				else:
					rand_max = 1.0
				data['random_input'] = np.random.uniform(low=0.0, high=rand_max, size=(s[0], P_run['G_model_params']['rand_size']))
		else:
			if 'multsteps' in P_run['model_name']:
				out_name = 'output_0'
			else:
				out_name = 'output'
			data = {'input_frames': X_flat[:,t_start:t+1]}
		data[out_name] = X[:,t+1].reshape((s[0], 1)+s[2:])

		yhat = model.predict(data, batch_size=P['batch_size'])[out_name]
		predictions[:,t] = yhat.reshape((s[0],)+s[2:])
		score_per_clip = ((yhat.reshape((s[0],np.prod(yhat.shape[1:]))) - data[out_name].reshape((s[0],np.prod(yhat.shape[1:]))))**2).mean(axis=-1)
		if P_run['is_GAN']:
			score = np.mean(score_per_clip)
		else:
			score = model.evaluate(data, batch_size=P['batch_size'])

		if 'ball' in P['data_file']:
			score *= 30*30
			score_per_clip *= 30*30
		print '  Score: '+str(score)
		print '  Score (from per clip): '+str(score_per_clip.mean())
		scores[t] = score
		scores_per_clip[:,t] = score_per_clip
	print 'Average Score: '+str(np.mean(scores))
	print 'STD Score: '+str(np.std(scores_per_clip.mean(axis=-1)))

	os.makedirs(P['save_dir'])

	f = open(P['save_dir']+'params.pkl','w')
	pkl.dump(P,f)
	f.close()

	f = open(P['save_dir']+'predictions.hkl','w')
	hkl.dump(predictions,f)
	f.close()

	spio.savemat(P['save_dir']+'predictions.mat', {'predictions': predictions[:5], 'X': X[:5]})

	f = open(P['save_dir']+'scores.hkl','w')
	hkl.dump(scores,f)
	f.close()



def test_training_val_error():

	run_num = 31
	base_dir = '/home/bill/Projects/Predictive_Networks/ball_GAN_runs_server/'
	predictions, actual_sequences = pkl.load(open(base_dir+'run_'+str(run_num)+'/predictions.pkl','r'))
	n = predictions.shape[0]
	s = np.prod(predictions.shape[2:])
	predictions = predictions[:,0].reshape((n,s))
	actual_sequences = actual_sequences[:,0].reshape((n,s))
	print 30*30*((predictions-actual_sequences)**2).mean()




def test_loss_fxn():

	params =  {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
	models = []

	models.append( initialize_model('facegen_rotation_prednet_twolayer', params) )
	models.append( initialize_model('facegen_rotation_prednet_twolayer_flattened', params) )

	f = open('/home/bill/Data/FaceGen_Rotations/clipset3/clipsval.hkl', 'r')
	X_val = hkl.load(f)
	f.close()
	X_flat_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]*X_val.shape[4]))
	data_val = {}
	data_val[0] = {'input_frames': X_flat_val[:,:5], 'output': X_val[:,5]}
	data_val[1] = {'input_frames': X_flat_val[:,:5], 'output': X_flat_val[:,5]}

	losses = {}

	for i in [0]:
		models[i] = initialize_weights('/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_9/model_best_weights.hdf5', models[i])

		#if i==0:
		data_val[i]['output'] = np.concatenate( (data_val[i]['output'], 2*np.ones_like(data_val[i]['output'])), axis=-1 )
		loss = {'output': 'weighted_mse'}
		# else:
		# 	loss = {'output': 'mse'}
		opt = RMSprop()

		# if i==1:
		# 	s = data_val['output'].shape
		# 	data_val['output'] = data_val['output'].reshape( (s[0], np.prod(s[1:])) )
		models[i].compile(optimizer=opt, loss=loss)
		losses[i] = models[i].evaluate(data_val[i], batch_size=6)

	for i in losses:
		print 'loss '+ str(i)+ ': '+str(losses[i])


# image set will be (n_ims, nx, ny)
# test_im will be (nx, ny)
def get_closest_image(test_im, image_set):

	n_ims = image_set.shape[0]
	test_im = test_im.flatten()
	image_set_flat = image_set.reshape( (n_ims, np.prod(image_set.shape[1:])) )
	distances = np.zeros(n_ims)
	for i in range(n_ims):
		distances[i] = np.linalg.norm((test_im-image_set_flat[i]))
	idx = np.argmin(distances)

	return image_set[idx]


def test_closest_ims():

	run_GAN = True

	if run_GAN:
		run_num = 342
		run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_'+str(run_num)+'/'
		base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'
		P = load_params(run_num, base_save_dir)
		P['n_evaluate'] = 40
		model = get_best_facegen_GAN_model()
		model = append_predict(model)
		predictions, actual_sequences,_ = evaluate_prednet(P, model, True)
	else:
		run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/'
		predictions, actual_sequences = pickle.load(open(run_dir+'predictions.pkl','r'))

	out_dir = run_dir + 'closest_im_plots/'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	P = pickle.load(open(run_dir+'params.pkl','r'))

	clips = hkl.load(open(P['training_files'][0],'r'))
	X = clips.reshape( (clips.shape[0]*clips.shape[1], clips.shape[3], clips.shape[4]))

	plt.figure()
	for i in range(predictions.shape[0]):
		print 'Im '+str(i)
		this_im = get_closest_image(predictions[i,0,0], X)

		plt.subplot(1,3,1)
		plt.imshow(actual_sequences[i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		d = np.linalg.norm(actual_sequences[i,0,0].flatten()-predictions[i,0,0].flatten())
		plt.title('Actual Image d='+str(np.round(100*d)/100))

		plt.subplot(1,3,2)
		plt.imshow(predictions[i,0,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.title('Generated Image')

		plt.subplot(1,3,3)
		plt.imshow(this_im, cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		d = np.linalg.norm(predictions[i,0,0].flatten()-this_im.flatten())
		plt.title('Closest Training Image d='+str(np.round(100*d)/100))

		plt.savefig(out_dir+'predicted_im_'+str(i)+'.jpg')


def analyze_share_encoder():

	start_run = 1
	end_run = 271
	exp = 'facegen_GAN'

	for r in range(start_run, end_run+1):
		try:
			P = load_params(r, base_save_dir = '/home/bill/Projects/Predictive_Networks/'+exp+'_runs/')
		except:
			print 'Couldnt load '+str(r)
			continue
		if 'share_encoder' in P['D_model_params']:
			if P['D_model_params']['share_encoder'] is False:
				print '*****not shared for '+str(r)


def get_best_facegen_GAN_model():

	# run_num = 307
	# model_str = 'G_model_weights_batch16500'
	# base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/'

	run_num = 342
	model_str = 'G_model_best_weights'
	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/'
	model = load_GAN_model_from_run(run_num, base_save_dir, model_str)

	return model


def get_best_facegen_MSE_model():

	run_num = 65
	base_save_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/'
	P = load_params(run_num, base_save_dir)
	model = initialize_model(P['model_name'], P['model_params'])
	model.load_weights(base_save_dir+'run_'+str(run_num)+'/model_best_weights.hdf5')

	return model


def create_denoising_clipset():

	P = {}
	P['orig_clipset'] = 13
	P['new_clipset'] = 17
	P['noise_val'] = 0.1


	base_dir = '/home/bill/Data/FaceGen_Rotations/clipset'
	old_dir = base_dir+str(P['orig_clipset'])+'/'
	new_dir = base_dir+str(P['new_clipset'])+'/'
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)

	for t in ['train', 'val','test']:
		X = hkl.load(open(old_dir+'clips'+t+'.hkl','r'))
		X = X[:,:2]
		n = X.shape[-1]*X.shape[-2]
		for i in range(X.shape[0]):
			idx = np.random.permutation(n)[:int(np.round(P['noise_val']*n))]
			r_idx = np.unravel_index(idx, X.shape[3:])
			X[i,1,0][r_idx] = 1.0
		hkl.dump(X, open(new_dir+'clips'+t+'.hkl','w'))

	pkl.dump(P, open(new_dir+'params.pkl','w'))




if __name__=='__main__':
	try:
		#test_loss_fxn()
		#create_denoising_clipset()
		# for i in range(40):
		#run_prednet()
		#run_prednet({'G_obj_weights': {'output': 1.0, 'pixel_output': 10.0}})
		#run_prednet({'G_obj_weights': {'output': 1.0, 'pixel_output': 20.0}})
		#make_plots_for_model(607, model_str = 'G_model_weights_batch13332')
		# for r in [661,662,663]:
		# 	make_plots_for_model(r)
		#run_prednet({'G_learning_rate': 0.005, 'D_learning_rate': 0.01})
		#run_prednet({'frames_in_params': 16})
		#run_prednet({'frames_in_params': [5, 15], 'frames_in_version': 'random'})
		#run_prednet({'model_params': {'batch_size': 8, 'nt_predict': 1, 'encoder_dropout': False, 'LSTM_dropout': False, 'nfilt': 16}, 'frames_in_params': 10})
		#run_prednet({'model_params': {'batch_size': 8, 'nt_predict': 1, 'encoder_dropout': False, 'LSTM_dropout': False, 'nfilt': 64}, 'frames_in_params': 10})
		#run_autoencoder()
		#test_caffe_model()
		#test_vgg()
		#finish_run(332)
		#create_mnist_shift_videos3()
		#plot_logs('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/facegen_runs/', [17, 18], ['32 filts', '64 filts'])

		#test_closest_ims()

		#save_predictions2mat(28)
		#check_random_weights(6, '/home/bill/Projects/Predictive_Networks/ball_GAN_runs_server/')

		#calculate_naive_error_rate()


		# poss_lr = [0.001, 0.0001, 0.00005, 0.00001, 0.000001]
		# poss_steps = [1,2]
		# n_runs = 200
		# for i in range(n_runs):
		# 	g_idx = np.random.randint(len(poss_lr))
		# 	d_idx = np.random.randint(len(poss_steps))
		# 	po = {'G_learning_rate': poss_lr[g_idx], 'nb_D_steps': poss_steps[d_idx]}
		# 	run_prednet(po)

		# for i in range(40):
		# 	r = np.random.rand()
		# 	if r<0.33:
		# 		w = 0.2
		# 	elif r<0.66:
		# 		w = 0.4
		# 	else:
		# 		w = 1.0
		# 	r = np.random.rand()
		# 	if r<0.33:
		# 		g = None
		# 		d = None
		# 	elif r<0.66:
		# 		g = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_400/G_model_best_weights.hdf5'
		# 		d = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_400/D_model_best_weights.hdf5'
		# 	else:
		# 		g = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_393/G_model_best_weights.hdf5'
		# 		d = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_393/D_model_best_weights.hdf5'
		# 	po = {}
		# 	po['G_obj_weights'] = {'output': w, 'pixel_output': 1.0}
		# 	po['G_initial_weights'] = g
		# 	po['D_initial_weights'] = d
		# 	run_prednet(po)

		# for i in range(40):
		# 	if np.random.rand()<0.5:
		# 		opt = 'rmsprop'
		# 		lr = 0.001
		# 		pw = 1.0
		# 		if np.random.rand()<0.5:
		# 			go = 0.1
		# 		else:
		# 			go = 0.01
		# 	else:
		# 		opt = 'sgd'
		# 		r = np.random.rand()
		# 		if r<0.33:
		# 			pw = 20.0
		# 			lr = 0.001
		# 			go = 1.0
		# 		elif r<0.66:
		# 			pw = 10.0
		# 			lr = 0.002
		# 			go = 1.0
		# 		else:
		# 			lr = 0.002
		# 			pw = 5.0
		# 			go = 1.0
		# 	po = {}
		# 	po['G_optimizer'] = opt
		# 	po['G_obj_weights'] = {'output': go, 'pixel_output': pw}
		# 	po['G_learning_rate'] = lr
		# 	run_prednet(po)

		# poss_lr = [0.001, 0.002, 0.005, 0.01, 0.02]
		#
		# n_runs = 200
		# for i in range(n_runs):
		# 	g_idx = np.random.randint(len(poss_lr))
		# 	d_idx = np.random.randint(len(poss_lr))
		# 	po = {'G_learning_rate': poss_lr[g_idx], 'D_learning_rate': poss_lr[d_idx]}
		# 	run_prednet(po)

		# n_runs = 100
		# #poss_GAN_weights = [1.0, 0.1, 0.01]
		# #poss_pixel_weights = [1.0, 10, 100]
		# for i in range(n_runs):
		# 	print 'Run '+str(i)
		# 	c = np.random.randint(4)
		# 	if c==0:
		# 		g_lr = 0.002
		# 		d_lr = 0.01
		# 	elif c==1:
		# 		g_lr = 0.005
		# 		d_lr = 0.01
		# 	elif c==2:
		# 		g_lr = 0.002
		# 		d_lr = 0.005
		# 	elif c==3:
		# 		g_lr = 0.005
		# 		d_lr = 0.02
		# 	# elif c==4:
		# 	# 	g_lr = 0.01
		# 	# 	d_lr = 0.01
		# 	#gi = np.random.randint(len(poss_GAN_weights))
		# 	#pi = np.random.randint(len(poss_pixel_weights))
		# 	po = {}
		# 	po['G_learning_rate'] = g_lr
		# 	po['D_learning_rate'] = d_lr
		# 	#po['G_obj_weights'] = {'output': poss_GAN_weights[gi], 'pixel_output': poss_pixel_weights[pi]}
		# 	po['G_obj_weights'] = {'output': 1.0, 'pixel_output': 0.0}
		# 	run_prednet(po)


		# n_runs = 100
		# poss_pairs = [(0.002, 0.01), (0.005, 0.01), (0.002, 0.005), (0.005, 0.02), (0.01, 0.01), (0.001, 0.005)]
		# for i in range(n_runs):
		# 	print 'Run '+str(i)
		# 	c = np.random.randint(len(poss_pairs))
		# 	po = {}
		# 	po['G_learning_rate'] = poss_pairs[c][0]
		# 	po['D_learning_rate'] = poss_pairs[c][1]
		#
		# 	if np.random.rand()<0.5:
		# 		m = 0.5
		# 	else:
		# 		m = 0.2
		# 	po['G_momentum'] = m
		# 	po['D_momentum'] = m
		#
		# 	if np.random.rand()<0.5:
		# 		use_batch_norm = True
		# 	else:
		# 		use_batch_norm = False
		# 	po['G_model_params'] = {'use_pixel_output': True, 'batch_size': 8, 'use_rand_input': False, 'use_batch_norm': use_batch_norm, 'rand_size': 128, 'encoder_fixed': False, 'decoder_fixed': False, 'n_LSTM': 512,  'pixel_output_flattened': False}
		# 	po['D_model_params'] = {'batch_size': 2*8, 'encoder_fixed': False, 'use_batch_norm': use_batch_norm, 'n_LSTM': 256}
		#
		# 	run_prednet(po)

		# n_runs = 100
		# poss_mom = [0.4, 0.2, 0.05, 0.01]
		# for i in range(n_runs):
		# 	print 'Run '+str(i)
		# 	c = np.random.randint(len(poss_mom))
		# 	po = {}
		# 	po['G_momentum'] = poss_mom[c]
		# 	po['D_momentum'] = poss_mom[c]
		#
		# 	if np.random.rand()<0.5:
		# 		G_nLSTM = 32*7*7
		# 	else:
		# 		G_nLSTM = 512
		# 	if np.random.rand()<0.5:
		# 		D_nLSTM = 256
		# 	else:
		# 		D_nLSTM = 512
		# 	if np.random.rand()<0.5:
		# 		po['frames_in_version'] = 'random'
		# 		po['frames_in_params'] = [5, 15]
		# 	else:
		# 		po['frames_in_version'] = 'fixed'
		# 		po['frames_in_params'] = 10
		# 	po['G_model_params'] = {'use_pixel_output': True, 'batch_size': 8, 'use_rand_input': False, 'use_batch_norm': False, 'rand_size': 128, 'encoder_fixed': False, 'decoder_fixed': False, 'n_LSTM': G_nLSTM,  'pixel_output_flattened': False}
		# 	po['D_model_params'] = {'batch_size': 2*8, 'encoder_fixed': False, 'use_batch_norm': False, 'n_LSTM': D_nLSTM}
		#
		# 	run_prednet(po)


		# n_runs = 100
		# poss_mom = [0.4, 0.2]
		# for i in range(n_runs):
		# 	print 'Run '+str(i)
		# 	c = np.random.randint(len(poss_mom))
		# 	po = {}
		# 	po['G_momentum'] = poss_mom[c]
		# 	po['D_momentum'] = poss_mom[c]
		#
		# 	if np.random.rand()<0.5:
		# 		po['frames_in_version'] = 'random'
		# 		po['frames_in_params'] = [5, 15]
		# 	else:
		# 		po['frames_in_version'] = 'fixed'
		# 		po['frames_in_params'] = 10
		#
		# 	run_prednet(po)




		#
		# n_runs = 100
		# for i in range(n_runs):
		# 	po = {}
		# 	if np.random.rand()<0.5:
		# 		po['G_optimizer'] = 'adam'
		# 		po['D_optimizer'] = 'adam'
		# 		po['G_learning_rate'] = 0.0002
		# 		if np.random.rand()<0.5:
		# 			po['D_learning_rate'] = 0.001
		# 		else:
		# 			po['D_learning_rate'] = 0.0002
		# 		po['G_momentum'] = 0.5
		# 		po['D_momentum'] = 0.5
		# 	else:
		# 		po['G_optimizer'] = 'sgd'
		# 		po['D_optimizer'] = 'sgd'
		# 		po['G_learning_rate'] = 0.002
		# 		po['D_learning_rate'] = 0.01
		# 		po['G_momentum'] = 0.5
		# 		po['D_momentum'] = 0.5
		# 	run_prednet(po)

		#print_learning_rates(50, 162)
		#make_results_spreadsheet(317, 330, 'facegen_GAN', False)
		#make_plots_for_last_model(3)
		#produce_prediction_for_dataset()

		# n_runs = 200
		# poss_pixel_weights = [1.0, 5.0]
		# poss_GAN_weights = [0.05, 0.01, 0.005, 0.001, 0.0001]
		# for i in range(n_runs):
		# 	po = {}
		# 	c = np.random.randint(2)
		# 	if c==0:
		# 		encoder_fixed = True
		# 	else:
		# 		encoder_fixed = False
		# 	c = np.random.randint(2)
		# 	if c==0:
		# 		share_RNN = True
		# 		RNN_params_fixed = encoder_fixed
		# 	else:
		# 		share_RNN = False
		# 		RNN_params_fixed = False
		# 	po['D_model_params'] = {'n_timesteps': 5, 'encoder_params_fixed': encoder_fixed, 'share_RNN': share_RNN, 'RNN_params_fixed': RNN_params_fixed}
		# 	gi = np.random.randint(len(poss_GAN_weights))
		# 	pi = np.random.randint(len(poss_pixel_weights))
		# 	po['G_obj_weights'] = {'output': poss_GAN_weights[gi], 'pixel_output': poss_pixel_weights[pi]}
		# 	run_prednet(po)

		# n_runs = 200
		# for i in range(n_runs):
		# 	po = {}
		# 	c = np.random.randint(2)
		# 	if c==0:
		# 		f_in = 'random'
		# 		f_in_p = [5, 15]
		# 	else:
		# 		f_in = 'fixed'
		# 		f_in_p = 10
		# 	po['frames_in_version'] = f_in
		# 	po['frames_in_params'] = f_in_p
		# 	c = np.random.randint(2)
		# 	if c==0:
		# 		po['G_optimizer'] = 'adam'
		# 		po['D_optimizer'] = 'adam'
		# 		poss_lr = [0.001, 0.002, 0.0005]
		# 		g_idx = np.random.randint(len(poss_lr))
		# 		d_idx = np.random.randint(len(poss_lr))
		# 		po['G_learning_rate'] = poss_lr[g_idx]
		# 		po['D_learning_rate'] = poss_lr[d_idx]
		# 	else:
		# 		po['G_optimizer'] = 'sgd'
		# 		po['D_optimizer'] = 'sgd'
		# 		poss_lr = [0.001, 0.002, 0.005, 0.01, 0.02]
		# 		g_idx = np.random.randint(len(poss_lr))
		# 		d_idx = np.random.randint(len(poss_lr))
		# 		po['G_learning_rate'] = poss_lr[g_idx]
		# 		po['D_learning_rate'] = poss_lr[d_idx]
		# 		poss_mom = [0.5, 0.9]
		# 		m_idx = np.random.randint(len(poss_mom))
		# 		po['G_momentum'] = poss_mom[m_idx]
		# 		po['D_momentum'] = poss_mom[m_idx]
		# 	run_prednet(po)
		#
		# poss_GAN_weights = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
		# poss_opt = ['rmsprop', 'sgd']
		#
		# done = [(0.001, 'rmsprop'), (0.0001, 'rmsprop'), (0.05, 'rmsprop'), (0.1, 'rmsprop'), (0.1, 'sgd'), (0.05, 'sgd')]
		#
		# all_pairs = []
		# for i in range(len(poss_GAN_weights)):
		# 	for j in range(len(poss_opt)):
		# 		tup = (poss_GAN_weights[i], poss_opt[j])
		# 		if tup not in done:
		# 			all_pairs.append((i,j))
		#
		# #run_idx = [1,5,9,0]
		# #run_idx = [2,6,10,13]
		# #run_idx = [3,7,11]
		# run_idx = [4,8,12]
		#
		# for r in run_idx:
		# 	po = {}
		# 	po['G_optimizer'] = poss_opt[all_pairs[r][1]]
		# 	if po['G_optimizer']=='sgd':
		# 		po['G_learning_rate'] = 0.01
		# 	else:
		# 		po['G_learning_rate'] = 0.001
		# 	po['G_obj_weights'] = {'output': poss_GAN_weights[all_pairs[r][0]], 'pixel_output': 1.0}
		# 	po['tag'] = 'r'+str(r)
		# 	print po['tag']
		# 	run_prednet(po)


		# while True:
		# 	po = {}
		# 	if np.random.randint(2)==0:
		# 		po['G_optimizer'] = 'rmsprop'
		# 		po['G_learning_rate'] = 0.001
		# 	else:
		# 		po['G_optimizer'] = 'sgd'
		# 		po['G_learning_rate'] = 0.01
		# 	if np.random.randint(2)==0:
		# 		po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False}
		# 	else:
		# 		po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': False, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False}
		# 	idx = np.random.randint(len(poss_GAN_weights))
		# 	po['G_obj_weights'] = {'output': poss_GAN_weights[idx], 'pixel_output': 1.0}
		# 	run_prednet(po)

		#produce_prediction_for_dataset()
		#test_training_val_error()
		save_final_predictions()

		#calculate_naive_error_rate()

		#make_results_spreadsheet(590, 606, 'facegen_GAN', False)

		#print_stuff(372, 377, 'facegen_GAN', False)


		# choices = [[452, False, 256],[479, False, 1024], [507, True, 1024]]
		# poss_w = [0.0001, 0.00005, 0.0003]
		# for i in range(40):
		# 	c = np.random.randint(len(choices))
		# 	w = np.random.randint(len(poss_w))
		# 	po = {}
		# 	po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': False, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False, 'use_fc_precat': choices[c][1], 'RNN_mult': 1.0, 'fc_precat_size': 1024, 'fusion_type': 'early', 'n_LSTM': choices[c][2]}
		# 	po['G_obj_weights'] = {'output': poss_w[w], 'pixel_output': 1.0}
		# 	po['D_initial_weights'] = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs/run_'+str(choices[c][0])+'/D_model_best_weights.hdf5'
		# 	run_prednet(po)

		#make_plots_for_model(607)
		# for r in range(590,607):
		# 	try:
		# 		make_plots_for_model(r)
		# 	except:
		# 		print 'no plots for '+str(r)


	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
