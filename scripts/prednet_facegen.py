import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model, load_vgg16_model, load_vgg16_test_model

from copy import deepcopy
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import hickle as hkl
import scipy as sp
import scipy.io as spio
import pickle as pkl
from scipy.misc import imread, imresize

from prednet import get_block_num_timesteps

if os.path.isdir('/home/bill/'):
	sys.path.append('/home/bill/Dropbox/Cox_Lab/General_Scripts/')
	import general_python_functions as gp
	sys.path.append('/home/bill/Libraries/keras/')
	from keras.datasets import mnist
	from keras.utils import np_utils
	from keras.models import standardize_X, slice_X
	from keras.optimizers import *
	sys.path.append('/home/bill/Libraries/keras/caffe/')
	from keras.caffe import convert

base_save_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/facegen_runs/'

def get_prednet_params(param_overrides=None):

	P = {}
	P['model_name'] = 'bouncing_ball_rnn_twolayer_multsteps'
	P['nt_train'] = 5
	P['model_params'] = {'n_timesteps': 5}
	P['initial_weights'] = None
	P['training_files'] = ['/home/bill/Data/FaceGen_Rotations/clipset0/clips.hkl']
	P['evaluation_file'] = '/home/bill/Data/FaceGen_Rotations/clipset0/clips.hkl'
	P['epochs_per_block'] = 10
	P['optimizer'] = 'rmsprop'
	P['n_evaluate'] = 20
	P['n_plot'] = 10
	P['save_model'] = True
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P



def run_prednet(param_overrides=None):

	P = get_prednet_params(param_overrides)

	print 'TRAINING MODEL'
	model= train_prednet(P)

	os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	if P['save_model']:
		model.save_weights(P['save_dir']+'model_weights.hdf5')

	print 'EVALUATING MODEL'
	for v in [True, False]:
		predictions, actual_sequences, pre_sequences = evaluate_prednet(P, model, v)

		if v:
			f = open(P['save_dir'] + 'predictions.pkl', 'w')
			pickle.dump([predictions, actual_sequences], f)
			f.close()

		print 'MAKING PLOTS'
		make_evaluation_plots(P, predictions, actual_sequences, pre_sequences, v)


def train_prednet(P):

	model = initialize_model(P['model_name'], P['model_params'])

	if P['initial_weights'] is not None:
		init_model = initialize_model(P['initial_weights'][0], P['initial_weights'][1])
		init_model.load_weights(P['initial_weights'][3])
		for layer in P['initial_weights'][2]:
			w = init_model.nodes[layer].get_weights()
			model.nodes[layer].set_weights(w)

	print 'Compiling'
	if P['model_name']=='bouncing_ball_rnn_twolayer_multsteps':
		loss = {}
		for t in range(P['model_params']['nt_predict']):
			loss['output_'+str(t)] = 'mse'
	else:
		loss={'output': 'mse'}
	model.compile(optimizer=P['optimizer'], loss=loss, obj_weights=P['obj_weights'])

	block_files = P['training_files']
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()
		n_ex = X.shape[0]
		n_batches = n_ex/128
		X = X[:(n_batches-1)*128] # 390 is max

		X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))

		for epoch in range(P['epochs_per_block']):
			nt = get_block_num_timesteps(P)
			data = {'input_frames': X_flat[:,:nt]} #, 'output': X[:,nt].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))}
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

	return model
