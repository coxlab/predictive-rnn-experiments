import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model, plot_model
from prednet import *

from copy import deepcopy
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import hickle as hkl
import scipy as sp
import scipy.io as spio
import pickle as pkl

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

#base_save_dir = '/home/bill/Projects/Predictive_Networks/mnist_GAN_runs/'
base_save_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/mnist_GAN_runs/'


def get_GAN_prednet_params(param_overrides=None):

	P = {}
	P['gen_model_name'] = 'mnist_prednet_G_to_D2'
	P['gen_model_params'] = {'use_prediction_output': True, 'prediction_output_weight': 0.0, 'use_feature_output': True, 'feature_output_weight': 10.0, 'use_LSTM': True, 'n_RNN_units': 512, 'rand_nx': 16} #{'use_prediction_output': True, 'prediction_output_weight': 20.0, 'use_LSTM': True, 'n_RNN_units': 512, 'rand_nx': 16}
	P['discrim_model_name'] = 'mnist_prednet_D2'
	P['discrim_model_params'] = {'use_LSTM': True, 'n_RNN_units': 512}
	P['feature_model_name'] = 'mnist_cnn2'
	P['feature_param_layers'] = ['conv0', 'conv1']
	P['decoder_model_name'] = 'mnist_deconv0'
	P['decoder_param_layers'] = ['_deconv0', '_deconv1']
	P['feature_level_prediction'] = True
	P['discrim_param_layers'] = ['D_RNN', 'D_FC0', 'D_FC1', 'D_FC2']#{'D_LSTM': 'D_LSTM', 'D_FC0': 'D_FC0', 'D_FC1': 'D_FC1', 'D_FC2': 'D_FC2', 'D_FC3': 'D_FC3'} #{'LSTM': 'LSTM1', 'FC0': 'FC1', 'FC1': 'FC2'}
	P['shared_weights'] = {'D_RNN': 'G_RNN'}#{'D_LSTM': 'LSTM0'}
	P['rand_width'] = 0.25
	P['rand_nx'] = 16
	P['nb_D_steps'] = 1
	P['nb_G_steps'] = 1
	P['adaptive_minibatch_tresh'] = 0 #0.65
	P['G_initial_weights'] = None #('mnist_prednet_G1', {'use_LSTM': True, 'n_RNN_units': 512, 'rand_nx': 16}, ['LSTM0', 'FC0', 'FC1', 'deconv0', 'deconv1'], '/home/bill/Projects/Predictive_Networks/mnist_GAN_runs/run_83/G_model_weights.hdf5') #('mnist_prednet_G0', ['LSTM0', 'FC0', 'deconv0', 'deconv1'], '/home/bill/Projects/Predictive_Networks/mnist_GAN_runs/run_26/G_model_weights.hdf5')
	P['pred_model_params'] = None
	P['training_files'] = ['/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/train_clips.hkl']
	P['evaluation_file'] = '/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/val_clips.hkl'
	P['mini_batch_size'] = 128
	P['epochs_per_block'] = 300 #3
	P['D_optimizer'] = RMSprop(1e-5)
	P['G_optimizer'] = RMSprop(0.002)
	P['n_evaluate'] = 128
	P['n_gen_per_eval'] = 5
	P['n_plot'] = 30
	P['save_model'] = True
	P['frame_orig_size'] = 28
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P

def get_generator_prednet_params(param_overrides=None):

	P = {}
	P['gen_model_name'] = 'mnist_prednet_G2' #'mnist_prednet_G1_v2'
	P['gen_model_params'] = {'use_prediction_output': True, 'prediction_output_weight': 0.00, 'use_LSTM': True, 'n_RNN_units': 512, 'rand_nx': 16} #{'use_prediction_output': True, 'prediction_output_weight': 20.0, 'use_LSTM': True, 'n_RNN_units': 512, 'rand_nx': 16}
	P['feature_model_name'] = 'mnist_cnn2'
	P['feature_param_layers'] = ['conv0', 'conv1']
	P['decoder_model_name'] = 'mnist_deconv0'
	P['decoder_param_layers'] = ['_deconv0', '_deconv1']
	P['feature_level_prediction'] = True
	P['nb_G_steps'] = 1
	P['rand_width'] = 0.25
	P['rand_nx'] = 16
	P['initial_weights'] = None
	P['training_files'] = ['/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/train_clips.hkl']
	P['evaluation_file'] = '/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/val_clips.hkl'
	P['mini_batch_size'] = 300*128
	P['epochs_per_block'] = 40 #3
	P['optimizer'] = 'rmsprop'
	P['n_evaluate'] = 128
	P['n_gen_per_eval'] = 3
	P['n_plot'] = 10
	P['save_model'] = True
	P['frame_orig_size'] = 28
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P

def load_params(run_num):
	f = open(base_save_dir + 'run_'+str(run_num)+'/params.pkl', 'r')
	p = pkl.load(f)
	return p

def run_GAN(param_overrides=None):

	P = get_GAN_prednet_params(param_overrides)

	os.mkdir(P['save_dir'])

	print 'TRAINING MODEL'
	G_model, D_model, G_model_gen_fxn, G_model_im_gen_fxn = train_GAN(P)

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	if P['save_model']:
		G_model.save_weights(P['save_dir']+'G_model_weights.hdf5')
		D_model.save_weights(P['save_dir']+'D_model_weights.hdf5')

	print 'EVALUATING MODEL'
	predictions, actual_sequences, outputs = evaluate_GAN(P, G_model, G_model_im_gen_fxn, D_model)

	f = open(P['save_dir'] + 'predictions.pkl', 'w')
	pickle.dump([predictions, actual_sequences], f)
	f.close()

	print 'MAKING PLOTS'
	make_evaluation_plots(P, predictions, actual_sequences)
	#make_evaluation_plots_outputs(P, outputs)



def run_generator(param_overrides=None):

	P = get_generator_prednet_params(param_overrides)

	print 'TRAINING MODEL'
	G_model= train_generator(P)

	os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	if P['save_model']:
		G_model.save_weights(P['save_dir']+'G_model_weights.hdf5')

	print 'EVALUATING MODEL'
	predictions, actual_sequences = evaluate_generator(P, G_model)

	f = open(P['save_dir'] + 'predictions.pkl', 'w')
	pickle.dump([predictions, actual_sequences], f)
	f.close()

	print 'MAKING PLOTS'
	make_evaluation_plots(P, predictions, actual_sequences)


def make_evaluation_plots(P, predictions, actual_sequences):

	t_str = 'GAN run_'+str(P['run_num'])
	out_dir = P['save_dir']+'plots/'
	os.mkdir(out_dir)

	plt.figure()
	for i in range(P['n_plot']):
		for j in range(predictions.shape[1]):
			plt.subplot(2, 2, 1)
			plt.imshow(actual_sequences[i,0,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
			plt.xlabel('Queue')
			plt.title(t_str + ' valclip_'+str(i) +' sample_'+str(j))
			plt.subplot(2, 2, 2)
			plt.imshow(actual_sequences[i,1,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
			plt.xlabel('Original')
			plt.subplot(2, 2, 3)
			plt.imshow(actual_sequences[i,2,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
			plt.xlabel('Actual Shift')
			plt.subplot(2, 2, 4)
			plt.imshow(predictions[i,j,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
			plt.xlabel('Sample Prediction')
			plt.savefig(out_dir+'valclip_'+str(i)+'_sample_'+str(j)+'.jpg')




def evaluate_GAN(P, G_model, G_model_gen_fxn, D_model):

	f = open(P['evaluation_file'], 'r')
	X = hkl.load(f)
	f.close()
	actual_sequences = X[:P['n_evaluate']]
	s = actual_sequences.shape

	predictions = np.zeros((P['n_evaluate'], P['n_gen_per_eval']) + s[2:])
	outputs = {}
	for l in ['G', 'D']:
		outputs[l] = np.zeros((P['n_evaluate'], P['n_gen_per_eval'], 2))
	for i in range(P['n_gen_per_eval']):
		input_frames = actual_sequences[:,:2]
		input_frames = input_frames.reshape((s[0], 2, 28*28))
		rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(P['n_evaluate'], 1, P['rand_nx'], P['rand_nx']))
		#rand_sample = np.zeros((P['n_evaluate'], 128))
		data = {'input_frames': input_frames, 'random_input': rand_sample}
		predictions[:,i] = extract_model_features(G_model_gen_fxn, G_model, data)
		outputs['G'][:,i] = G_model.predict(data)['output']
		#data = {'previous_frames': input_frames, 'proposed_frames': predictions[:,i]}
		#outputs['D'][:,i] = D_model.predict(data)['output']

	return predictions, actual_sequences, outputs



def evaluate_generator(P, G_model):

	f = open(P['evaluation_file'], 'r')
	X = hkl.load(f)
	f.close()
	actual_sequences = X[:P['n_evaluate']]
	s = actual_sequences.shape

	predictions = np.zeros((P['n_evaluate'], P['n_gen_per_eval']) + s[2:])
	for i in range(P['n_gen_per_eval']):
		input_frames = actual_sequences[:,:2]
		input_frames = input_frames.reshape((s[0], 2, 28*28))
		rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(P['n_evaluate'], 1, P['rand_nx'], P['rand_nx']))
		#rand_sample = np.zeros((P['n_evaluate'], 128))
		data = {'input_frames': input_frames, 'random_input': rand_sample}
		if P['feature_level_prediction']:
			yhat = G_model.predict(data)['prediction_output']
		else:
			yhat = G_model.predict(data)['output']
		predictions[:,i] = yhat

	return predictions, actual_sequences


def copy_params(w):
	w2 = []
	for v in w:
		w2.append(np.copy(v))
	return w2

def train_GAN(P):

	feature_model = load_model(P['feature_model_name'])
	feature_gen_fxn = create_feature_fxn(feature_model, 'pool1')
	if P['decoder_model_name'] is not None:
		decoder_model = load_model(P['decoder_model_name'])

	D_model = initialize_model(P['discrim_model_name'], P['discrim_model_params'])
	plot_model(D_model)
	for layer in P['feature_param_layers']:
		w = feature_model.nodes[layer].get_weights()
		if not P['feature_level_prediction']:
			D_model.nodes['proposed_'+layer].set_weights(copy_params(w))
		D_model.nodes['previous_'+layer].set_weights(copy_params(w))
	print 'Compiling Discriminator'
	#D_model.compile(optimizer=P['D_optimizer'], loss={'output': 'GAN_discriminator_loss'})

	G_model = initialize_model(P['gen_model_name'], P['gen_model_params'])
	plot_model(G_model)
	for layer in P['feature_param_layers']:
		w = feature_model.nodes[layer].get_weights()
		G_model.nodes['G_'+layer].set_weights(copy_params(w))
		#G_model.nodes['D_'+layer].set_weights(copy_params(w))
	if P['decoder_model_name'] is not None:
		for layer in P['decoder_param_layers']:
			w = decoder_model.nodes[layer].get_weights()
			G_model.nodes['decoder'+layer].set_weights(copy_params(w))
	print 'Compiling Generator'
	l_dict = {'output': 'GAN_generator_loss'}
	obj_weights = {'output': 1.0}
	if P['gen_model_params']['use_prediction_output']:
		l_dict['prediction_output'] = 'mse'
		obj_weights['prediction_output'] = P['gen_model_params']['prediction_output_weight']
	if P['gen_model_params']['use_feature_output']:
		l_dict['feature_output'] = 'mse'
		obj_weights['feature_output'] = P['gen_model_params']['feature_output_weight']

	G_model.compile(optimizer=P['G_optimizer'], loss=l_dict, obj_weights=obj_weights)

	#G_model_gen_fxn = create_feature_fxn(G_model, 'deconv_satlu')
	G_model_gen_fxn = create_feature_fxn(G_model, 'G_proposed_features')
	G_model_im_gen_fxn = create_feature_fxn(G_model, 'decoder_deconv_output')

	if P['G_initial_weights'] is not None:
		init_model = initialize_model(P['G_initial_weights'][0], P['G_initial_weights'][1])
		init_model.load_weights(P['G_initial_weights'][3])
		for layer in P['G_initial_weights'][2]:
			w = init_model.nodes[layer].get_weights()
			G_model.nodes[layer].set_weights(copy_params(w))

	# D_fxns = {}
	# for layer in ['flatten1', 'LSTM', 'expand_time', 'FC0', 'FC1']:
	# 	D_fxns[layer] = create_feature_fxn(D_model, layer)
	# G_fxns = {}
	# for layer in ['flatten1', 'LSTM1', 'previous_features', 'FC1', 'FC2']:
	# 	G_fxns[layer] = create_feature_fxn(G_model, layer)

	for layer in P['discrim_param_layers']:
		w = D_model.nodes[layer].get_weights()
		#G_model.nodes[P['discrim_param_layers'][layer]].set_weights(copy_params(w)) #when it was dictionary
		#G_model.nodes[layer].set_weights(copy_params(w))
		G_model.nodes[layer].set_weights(w)

	for layer in P['shared_weights']:
		w = G_model.nodes[P['shared_weights'][layer]].get_weights()
		#D_model.nodes[layer].set_weights(copy_params(w))
		D_model.nodes[layer].set_weights(w)

	plt_dir = P['save_dir'] + 'intermediate_plots/'
	os.mkdir(plt_dir)

	# format validation data
	f = open(P['evaluation_file'], 'r')
	X = hkl.load(f)
	f.close()
	X_input = X[:,:2]
	X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 28*28))
	X_predict = X[:,2]
	input_frames = X_input[:128]
	rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(128, 1, P['rand_nx'], P['rand_nx']))
	y_val = np.zeros((128,2), int)
	y_val[:,1] = 1
	data_vis = {'input_frames': input_frames, 'random_input': rand_sample, 'output': y_val}
	data_vis['prediction_output'] = X_predict[:128]

	plt.figure()
	block_files = P['training_files']
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()
		X = X[:390*128] # 390 is max

		X_input = X[:,:2]
		X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 28*28))
		X_predict = X[:,2]
		if P['feature_level_prediction']:
			X_predict_frames = np.copy(X_predict)
			X_predict = extract_model_features(feature_gen_fxn, feature_model, {'input': X_predict})

		n_ex = P['mini_batch_size']
		y_D = np.zeros((2*n_ex,2), int)
		y_D[:n_ex,1] = 1
		y_D[n_ex:,0] = 1

		y_G = np.zeros((n_ex,2), int)
		y_G[:,1] = 1

		if P['feature_level_prediction']:
			pname = 'proposed_features'
		else:
			pname = 'proposed_frames'

		data_D = {'previous_frames': [], pname: []}
		data_D['output'] = y_D

		data_G = {'input_frames': [], 'random_input': [], 'output': y_G}

		for epoch in range(P['epochs_per_block']):
			print 'Starting epoch '+str(epoch)
			print ''
			idxs_G = get_mini_batch_idxs(X_input.shape[0], P['mini_batch_size'])
			idxs_D = get_mini_batch_idxs(X_input.shape[0], P['mini_batch_size'])

			for mini_batch in range(len(idxs_G)):
				#print 'Starting minibatch '+str(mini_batch)
				idx_G = idxs_G[mini_batch]
				idx_D = idxs_D[mini_batch]
				#n_ex = len(idx_G)
				print ''
				print 'Training Discriminator'
				print ''

				# copy shared weights
				for layer in P['shared_weights']:
					w = G_model.nodes[P['shared_weights'][layer]].get_weights()
					#D_model.nodes[layer].set_weights(copy_params(w))
					D_model.nodes[layer].set_weights(w)

				for d_step in range(P['nb_D_steps']):
					input_frames = X_input[idx_D]
					rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(n_ex, 1, P['rand_nx'], P['rand_nx']))
					#rand_sample = np.zeros((n_ex, 128))
					data_tmp = {'input_frames': input_frames, 'random_input': rand_sample}
					#yhat_G = G_model.predict(data)['output'][:,1]
					# extract features
					X_generated = extract_model_features(G_model_gen_fxn, G_model, data_tmp)
					#for k in range(5):
					#	plt.imshow(X_generated[k,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
					#	plt.savefig(plt_dir + 'im'+str(k)+'_epoch'+str(epoch)+'_batch'+str(mini_batch) +'_preDtrain.jpg')

					# G_feats = {}
					# for layer in G_fxns:
					# 	G_feats[layer] = extract_model_features(G_fxns[layer], G_model, data)

					previous_frames = np.vstack((input_frames, input_frames))
					proposed_frames = np.vstack((X_predict[idx_D], X_generated))

					data_D['previous_frames'] = previous_frames
					data_D[pname] =  proposed_frames

					# D_feats = {}
					# for layer in D_fxns:
					# 	D_feats[layer] = extract_model_features(D_fxns[layer], D_model,  {'previous_frames': input_frames, 'proposed_frames': X_generated})

					#if sp.stats.pearsonr(yhat_D, yhat_G)<0.99:
					#	pdb.set_trace()

					# if yhat_D.mean()>0.95 or yhat_D.mean()<0.05:
					# 	plt.scatter(yhat_D, yhat_G)
					# 	plt.show(block=False)
					# 	pdb.set_trace()

					yhat = D_model.predict(data_D)['output']
					D_score = (yhat[:n_ex,1].mean() + 1.-yhat[n_ex:,1].mean())/2
					print 'D prob when true: '+str(yhat[:n_ex,1].mean())
					print 'D prob when false: '+str(yhat[n_ex:,1].mean())
					if P['adaptive_minibatch_tresh']==0:
						D_model_good = False
					else:
						D_model_good = D_score > P['adaptive_minibatch_tresh']
					while not D_model_good:
						D_model.fit(data_D, batch_size=128, nb_epoch=1, verbose=1)
						yhat = D_model.predict(data_D)['output']
						D_score = (yhat[:n_ex,1].mean() + 1.-yhat[n_ex:,1].mean())/2
						print 'D prob when true: '+str(yhat[:n_ex,1].mean())
						print 'D prob when false: '+str(yhat[n_ex:,1].mean())
						if P['adaptive_minibatch_tresh']==0:
							D_model_good = True
						else:
							D_model_good = D_score > P['adaptive_minibatch_tresh']
					#yhat_D = yhat[n_ex:,1]
					# print 'MEAN PROB true: '+str(yhat[:n_ex,1].mean())
					# print 'STD PROB true: '+str(yhat[:n_ex,1].std())
					# print 'MEAN PROB generated: '+str(yhat[n_ex:,1].mean())
					# print 'STD PROB generated: '+str(yhat[n_ex:,1].std())

				#print 'Copying Discriminator weights'
				for layer in P['discrim_param_layers']:
					w = D_model.nodes[layer].get_weights()
					G_model.nodes[layer].set_weights(w)

				# copy shared weights
				for layer in P['shared_weights']:
					w = D_model.nodes[layer].get_weights()
					G_model.nodes[P['shared_weights'][layer]].set_weights(w)

				print ''
				print 'Training Generator'
				print ''

				for g_step in range(P['nb_G_steps']):
					input_frames = X_input[idx_G]
					rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(n_ex, 1, P['rand_nx'], P['rand_nx']))

					data_G['input_frames'] = input_frames
					data_G['random_input'] = rand_sample
					if P['gen_model_params']['use_prediction_output']:
						if P['feature_level_prediction']:
							data_G['prediction_output'] = X_predict_frames[idx_G]
						else:
							data_G['prediction_output'] = X_predict[idx_G]
					if P['feature_level_prediction']:
						if P['gen_model_params']['use_feature_output']:
							data_G['feature_output'] = X_predict[idx_G]
					#pdb.set_trace()
					yhat = G_model.predict(data_G)['output']
					G_score = yhat[:,1].mean()
					print 'G score: ' + str(G_score)
					if P['adaptive_minibatch_tresh']==0:
						G_model_good = False
					else:
						G_model_good = G_score > P['adaptive_minibatch_tresh']
					while not G_model_good:
						G_model.fit(data_G, batch_size=128, nb_epoch=1, verbose=1)
						yhat = G_model.predict(data_G)['output']
						G_score = yhat[:,1].mean()
						print 'G score: ' + str(G_score)
						if P['adaptive_minibatch_tresh']==0:
							G_model_good = True
						else:
							G_model_good = G_score > P['adaptive_minibatch_tresh']

					if mini_batch==0: #or mini_batch%100==0:
						X_generated = G_model.predict(data_vis)['prediction_output']
						for k in range(2):
							plt.imshow(X_generated[k,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
							plt.savefig(plt_dir + 'im'+str(k)+'_epoch'+str(epoch)+'_batch'+str(mini_batch)+'_postGtrain.jpg')
					# pdb.set_trace()

					#if yhat[:,1].std()>0.4:
					# 	pdb.set_trace()
					#print 'MEAN PROB: '+str(yhat[:,1].mean())
					#print 'STD PROB: '+str(yhat[:,1].std())

	return G_model, D_model, G_model_gen_fxn, G_model_im_gen_fxn


def train_generator(P):

	feature_model = load_model(P['feature_model_name'])
	feature_gen_fxn = create_feature_fxn(feature_model, 'pool1')
	if P['decoder_model_name'] is not None:
		decoder_model = load_model(P['decoder_model_name'])

	G_model = initialize_model(P['gen_model_name'], P['gen_model_params'])
	for layer in P['feature_param_layers']:
		w = feature_model.nodes[layer].get_weights()
		G_model.nodes['G_'+layer].set_weights(copy_params(w))
	if P['decoder_model_name'] is not None:
		for layer in P['decoder_param_layers']:
			w = decoder_model.nodes[layer].get_weights()
			G_model.nodes['decoder'+layer].set_weights(copy_params(w))
	l_dict = {'output': 'mse'}
	obj_weights = {'output': 1.0}
	if P['feature_level_prediction']:
		l_dict['prediction_output'] = 'mse'
		obj_weights['prediction_output'] = P['gen_model_params']['prediction_output_weight']
	print 'Compiling Generator'
	G_model.compile(optimizer=P['optimizer'], loss=l_dict)

	#G_model_im_gen_fxn = create_feature_fxn(G_model, 'decoder_deconv_output')

	block_files = P['training_files']
	for block in range(len(block_files)):
		f = open(block_files[block], 'r')
		X = hkl.load(f)
		f.close()
		X = X[:300*128] # 390 is max

		X_input = X[:,:2]
		X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 28*28))
		X_predict = X[:,2]
		X_predict = X_predict.reshape((X_input.shape[0], 1, 28, 28))
		if P['feature_level_prediction']:
			X_predict_frames = np.copy(X_predict)
			X_predict = extract_model_features(feature_gen_fxn, feature_model, {'input': X_predict})

		for epoch in range(P['epochs_per_block']):
			print 'Starting epoch '+str(epoch)
			idxs_G = get_mini_batch_idxs(X_input.shape[0], P['mini_batch_size'])
			for mini_batch in range(len(idxs_G)):
				print 'Starting minibatch '+str(mini_batch)
				idx_G = idxs_G[mini_batch]
				n_ex = len(idx_G)

				for g_step in range(P['nb_G_steps']):
					input_frames = X_input[idx_G]
					rand_sample = np.random.uniform(low=-P['rand_width'], high=P['rand_width'], size=(n_ex, 1, P['rand_nx'], P['rand_nx']))
					#rand_sample = np.zeros((n_ex, 128))
					data = {'input_frames': input_frames, 'random_input': rand_sample, 'output': X_predict[idx_G]}
					if P['feature_level_prediction']:
						data['prediction_output'] = X_predict_frames[idx_G]
					G_model.fit(data, batch_size=128, nb_epoch=1, verbose=1)

	return G_model

def train_deconv_net():

	feat_model = load_model('mnist_cnn2')
	feat_fxn = create_feature_fxn(feat_model, 'pool1')

	(X_orig, y_orig), _ = load_mnist()
	X = X_orig[:50000]
	X_val = X_orig[50000:]
	data = {'input': X}

	features = extract_model_features(feat_fxn, feat_model, data)
	val_features = extract_model_features(feat_fxn, feat_model, {'input': X_val})

	model = initialize_model('mnist_deconv0')
	model.compile(optimizer='rmsprop', loss={'output': 'mse'})
	data = {'input': features, 'output': X}
	model.fit(data, validation_data = {'input': val_features, 'output': X_val}, nb_epoch=125, verbose=1)
	X_hat = model.predict({'input': features})['output']
	X_val_hat = model.predict({'input': val_features})['output']
	l = model.evaluate({'input': val_features, 'output': X_val})
	print l

	for i in range(30):
		plt.subplot(1, 2, 1)
		plt.imshow(X[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Original')

		plt.subplot(1, 2, 2)
		plt.imshow(X_hat[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Reconstructed')

		plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/mnist_deconv0_plots/mse_maxunpooling/125_epochs_nodropout/im_'+str(i)+'.jpg')

		plt.subplot(1, 2, 1)
		plt.imshow(X_val[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Original')

		plt.subplot(1, 2, 2)
		plt.imshow(X_val_hat[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Reconstructed')

		plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/mnist_deconv0_plots/mse_maxunpooling/125_epochs_nodropout/val_im_'+str(i)+'.jpg')

	model.save_weights('/home/bill/Projects/Predictive_Networks/models/'+'mnist_deconv0'+'_weights0.hdf5')





def get_mini_batch_idxs(n_ex, mini_batch_size, permute=False):

	if permute:
		all_idx = np.random.permutation(n_ex)
	else:
		all_idx = np.array([i for i in range(n_ex)])
	idxs = []
	n_blocks = int(np.ceil(float(n_ex)/mini_batch_size))
	for i in range(n_blocks):
		i_start = i*mini_batch_size
		if i==n_blocks-1:
			i_end = n_ex
		else:
			i_end = (i+1)*mini_batch_size
		idxs.append(all_idx[i_start:i_end])

	return idxs


def generate_rotation_clip_set2(noise_std):

	n_frames = 11
	base_folder = '/home/bill/Data/MNIST/Rotation_Clips/clip_set2/'
	if not os.path.exists(base_folder):
		os.mkdir(base_folder)

	n_train = 50000

	(X_orig, y_orig), _ = load_mnist()
	y_orig = np.argmax(y_orig, axis=-1)
	X = {}
	y = {}
	X['train'] = X_orig[:n_train]
	y['train'] = y_orig[:n_train]
	X['val'] = X_orig[n_train:]
	y['val'] = y_orig[n_train:]
	nx = X_orig.shape[-1]

	count = 0
	for t in ['train', 'test']:
		n_ims = X[t].shape[0]
		clip_frames = np.zeros((n_ims, n_frames, 1, nx, nx)).astype(np.float32)
		actual_thetas = np.zeros((n_ims, n_frames))
		rotation_info = {'center_y': [], 'center_x':[], 'theta0':[], 'angular_speed':[]}
		for i in range(n_ims):
			center_x = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
			center_y = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
			theta0 = 0.0
			angular_speed = np.random.uniform(-np.pi/6, np.pi/6)
			clip_frames[i,:,0], actual_thetas[i] = create_rotation_clip2(X[t][i,0], center_x, center_y, theta0, angular_speed, n_frames, noise_std)
			rotation_info['center_x'].append(center_x)
			rotation_info['center_y'].append(center_y)
			rotation_info['theta0'].append(theta0)
			rotation_info['angular_speed'].append(angular_speed)
			if i%100==0:
				print 'Done clip '+str(i+1)+'/'+str(n_ims)
		fname = base_folder + 'mnist_rotations_'+t+'_noise'+str(noise_std)+'.hkl'
		hkl.dump(clip_frames, fname, mode='w')
		idxs = np.array([i+count for i in range(n_ims)]).astype(int)
		fname = base_folder + 'mnist_rotations_'+t+'_noise'+str(noise_std)+'.hkl'
		f = open(fname, 'w')
		pkl.dump([idxs, rotation_info, actual_thetas], f)
		f.close()



def create_rotation_clip2(orig_im, center_x, center_y, theta0, angular_speed, n_frames, noise_std = 0):

	clip = np.zeros((n_frames, orig_im.shape[0], orig_im.shape[1])).astype(np.float32)
	theta = theta0
	actual_thetas = np.zeros(n_frames)
	for i in range(n_frames):
		theta += i*angular_speed + noise_std*np.random.normal()
		clip[i] = rotate_image(orig_im, theta, center_x, center_y)
		actual_thetas[i] = theta

	return clip, actual_thetas


def create_mnist_shift_videos():

	out_dir = '/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/'
	n_train = 50000

	(X_train, _), _ = load_mnist()
	n_ex = X_train.shape[0]
	frame_size = X_train.shape[-1]
	shapes = ['circle', 'triangle', 'square']

	X = np.zeros((n_ex, 3, 1, frame_size, frame_size)).astype(np.float32)
	clip_info = []

	for i in range(n_ex):
		shape_idx = np.random.randint(len(shapes))
		shape = shapes[shape_idx]
		shift_x = np.random.randint(-5, 6)
		shift_y = np.random.randint(-5, 6)
		theta = np.random.uniform(-np.pi/4, np.pi/4)
		scale = np.random.uniform(0.5, 1.5)
		im = create_shape_frame(shape, frame_size, shift_x, shift_y, scale, theta)
		X[i,0,0] = im
		X[i,1] = X[i]
		clip_dict = {'shape': shape, 'shape_shift_x': shift_x, 'shape_shift_y': shift_y, 'scale': scale, 'theta': theta}

		if shape=='circle':
			shift_x = np.random.randint(-7, -2)
			shift_y = np.random.randint(-3, 4)
		elif shape=='square':
			shift_x = np.random.randint(3, 8)
			shift_y = np.random.randint(-3, 4)
		elif shape=='triangle':
			if np.random.rand()<0.5:
				shift_x = np.random.randint(3, 8)
			else:
				shift_x = np.random.randint(-7, -2)
			shift_y = np.random.randint(-3, 4)

		X[i,2,0] = translate_im(X[i,0], shift_x, shift_y)
		clip_dict['num_shift_x'] = shift_x
		clip_dict['num_shift_y'] = shift_y
		clip_info.append(clip_dict)
		pdb.set_trace()

	X_train = X[:n_train]
	X_val = X[n_train:]

	f = open(out_dir + 'train_clips.hkl', 'w')
	hkl.dump(X_train, f)
	f.close()
	f = open(out_dir + 'val_clips.hkl', 'w')
	hkl.dump(X_val, f)
	f.close()



def create_feature_fxn(model, layer_name):

	ins = [model.inputs[name].input for name in model.input_order]
	output = model.nodes[layer_name]
	ys_test = [output.get_output(False)]
	pred_fxn = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True, on_unused_input='warn')

	return pred_fxn


def extract_model_features(feature_fxn, model, X, batch_size = 128):

	ins = [X[name] for name in model.input_order]
	features = model._predict_loop(feature_fxn, ins, batch_size = batch_size)[0]

	return features


def extract_model_features_unevenX(feature_fxn, model, X, batch_size = 50):

	n_exs = {}
	for val in X:
		n_exs[val] = X[val].shape[0]

	min_ex = np.min(n_exs.values())

	idxs = {}
	for val in X:
		idxs[val] = get_mini_batch_idxs(n_exs[val], n_exs[val]/min_ex * batch_size, permute=False)
		n_batches = len(idxs[val])

	for i in range(n_batches):
		ins_batch = [X[name][idxs[name][i]] for name in model.input_order]
		batch_outs = feature_fxn(*ins_batch)
		if i==0:
			features = np.zeros((min_ex,) + batch_outs[0].shape[1:])
			counter = 0
		b_size = batch_outs[0].shape[0]
		features[counter:counter+b_size] = batch_outs[0]
		counter += b_size

	if counter != min_ex:
		print 'In extract features, something is messed up'
		pdb.set_trace()

	return features


def plot_shift_clips():

	n_plot = 50
	f = open('/home/bill/Data/MNIST/MNIST_Shift_Clips/clipset_1/val_clips.hkl','r')
	clips = hkl.load(f)
	f.close()
	save_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/shift_plots_clipset_1/'

	plt.figure()
	for i in range(n_plot):
		plt.subplot(1, 3, 1)
		plt.imshow(clips[i,0,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Indicator')
		plt.subplot(1, 3, 2)
		plt.imshow(clips[i,1,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Original')
		plt.subplot(1, 3, 3)
		plt.imshow(clips[i,2,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Shifted')
		plt.savefig(save_dir + 'val_clip_'+str(i)+'.jpg')


def create_mnist_shift_videos2():

	out_dir = '/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/'
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
		for i in [0,1,2]:
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

			if j==0:
				shift_x = np.random.randint(-2, 3)
				shift_y = np.random.randint(-10, -2)
			elif j==1:
				shift_x = np.random.randint(-2, 3)
				shift_y = np.random.randint(3, 11)
			elif j==2:
				if np.random.rand()<0.5:
					shift_y = np.random.randint(3, 11)
				else:
					shift_y = np.random.randint(-10, -2)
				shift_x = np.random.randint(-2, 3)

			clips[i,2,0] = translate_im(clips[i,1,0], shift_x, shift_y)
			clip_dict['num_shift_x'] = shift_x
			clip_dict['num_shift_y'] = shift_y
			clip_info.append(clip_dict)

		f = open(out_dir + t+'_clips.hkl', 'w')
		hkl.dump(clips, f)
		f.close()
		f = open(out_dir + t+'_clip_info.pkl', 'w')
		pkl.dump(clip_info, f)
		f.close()


def translate_im(im, dx, dy):
	n = im.shape[0]
	new_im = np.zeros((n, n)).astype(np.float32)

	for i in range(n):
		for j in range(n):
			x = i - dx
			y = j - dy
			if x<0 or x>im.shape[0]-1 or y<0 or y>im.shape[1]-1:
				val = 0
			else:
				val = im[x, y]
			new_im[i,j] = val

	return new_im


def create_shape_frame(shape_name, frame_size, shift_x, shift_y, scale, theta, threshold = 0.01):

	im_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/images/'

	im_file = im_dir + shape_name +'_' + str(frame_size) + '.mat'
	im_loaded = spio.loadmat(im_file)['m'].astype(np.float32)

	# Rotate image
	R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

	im = np.zeros((frame_size, frame_size)).astype(np.float32)
	center = int(np.round(float(im.shape[0])/2))
	for i in range(im.shape[0]):
		dx = i-center
		for j in range(im.shape[1]):
			dy = j-center
			dv = np.array([[dx], [dy]])
			v = np.dot(R, dv)
			x = int(np.round(center+v[0]))
			y = int(np.round(center+v[1]))
			if x<0 or x>im.shape[0]-1 or y<0 or y>im.shape[1]-1:
				val = 0
			else:
				val = im_loaded[x, y]
			im[i,j] = val

	pdb.set_trace()

	# Resize Image
	im = scipy.ndimage.zoom(im, scale)
	im[im<threshold] = 0

	if im.shape[0] != frame_size:

		d = np.abs(im.shape[0] - frame_size)
		if d%2==0:
			dl = d/2
			dr = dl
			du = dl
			dd = dl
		else:
			if np.random.uniform()<0.5:
				dl = d/2+1
				dr = d/2
			else:
				dl = d/2+1
				dr = d/2
			if np.random.uniform()<0.5:
				du = d/2+1
				dd = d/2
			else:
				du = d/2+1
				dd = d/2
		if im.shape[0]>frame_size:
			im = im[dd:du,dl:dr]
		else:
			im2 = np.zeros((frame_size, frame_size)).astype(np.float32)
			im2[dd:du,dl:dr] = im
			im = im2

	im = translate_im(im, shift_x, shift_y)

	return im





if __name__=='__main__':
	try:
		#create_mnist_shift_videos2()
		plot_shift_clips()
		#run_GAN()
		#train_deconv_net()
		#run_generator()

		#generate_rotation_clip_set2(0.05)

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
