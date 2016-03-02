import theano
import pdb, sys, traceback, os
from keras_models import load_model, initialize_model, plot_model, initialize_GAN_models
from prednet import load_mnist, extract_features_graph

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import hickle as hkl
import scipy as sp
import scipy.io as spio
import pickle as pkl
import pandas as pd

from prednet_utils import load_prednet_model, get_prednet_predictions, get_data_size
from prednet2 import append_predict, initialize_weights, plot_error_log

def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *

cname = get_computer_name()

sys.path.append(get_scripts_dir() +'General_Scripts/')

import general_python_functions as gp
sys.path.append('/home/bill/Libraries/keras/')
from keras.utils import np_utils
from keras.optimizers import *

project_dir = '/home/bill/Projects/Predictive_Networks/'

def get_refinenet_params(param_overrides=None):

	P = {}
	P['version'] = 'facegen'

	if P['version']=='facegen':
		base_save_dir = project_dir + 'facegen_refine_runs/'
		P['train_file'] = '/home/bill/Data/RefineNet_Data/Facegen_Rotations/set_0/data_train.hkl'
		P['val_file'] = '/home/bill/Data/RefineNet_Data/Facegen_Rotations/set_0/data_val.hkl'
		P['G_model_name'] = 'facegen_refine_G'
		P['G_model_params'] = None
		P['D_model_name'] = 'facegen_refine_D'
		P['D_model_params'] = {'use_batch_norm': False}
		P['G_loss'] = {'output': 'my_bce', 'pixel_output': 'mse'}
		P['G_obj_weights'] = {'output': 0.00, 'pixel_output': 1.0}
		P['D_loss'] = {'output': 'my_bce_pos'}
		P['G_initial_weights'] = base_save_dir+'run_74/G_model_weights.hdf5'
		P['D_initial_weights'] = base_save_dir+'run_74/D_model_weights.hdf5'
		P['G_optimizer'] = 'sgd'
		P['D_optimizer'] = 'sgd'
		P['G_learning_rate'] = 0.01
		P['D_learning_rate'] = 0.01
		P['G_momentum'] = 0.4
		P['D_momentum'] = 0.5

		P['batch_size'] = 32
		P['same_batches'] = True
		P['num_batches'] = 5 * 4000/P['batch_size']
		P['plot_frequency'] = 25
		P['print_interval'] = 25
		P['val_frequency'] = 25
		P['model_save_frequency'] = None #4000/P['batch_size']
		P['nb_D_steps'] = 1
		P['nb_G_steps'] = 1
		P['nb_D_pre_epochs'] = 0
		P['n_eval_int_plot'] = 3
		P['kill_criteria'] = None

	P['n_evaluate'] = 10
	P['n_plot'] = 10

	P['tag'] = ''
	P['save_model'] = True
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P


def run_refinenet(param_overrides=None):

	P = get_refinenet_params(param_overrides)

	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pkl.dump(P, f)
	f.close()

	print 'Save dir '+str(P['save_dir'])
	print 'TRAINING MODEL'
	G_model, D_model, log, G_best_weights, D_best_weights = train_refinenet(P)

	f = open(P['save_dir'] + 'log.pkl', 'w')
	pkl.dump(log, f)
	f.close()

	if P['save_model']:
		G_model.save_weights(P['save_dir']+'G_model_weights.hdf5')
		D_model.save_weights(P['save_dir']+'D_model_weights.hdf5')
		if G_best_weights is not None:
			G_model.set_weights(G_best_weights)
			D_model.set_weights(D_best_weights)
			G_model.save_weights(P['save_dir']+'G_model_best_weights.hdf5',overwrite=True)
			D_model.save_weights(P['save_dir']+'D_model_best_weights.hdf5',overwrite=True)

	plot_error_log(P, log)

	print 'EVALUATING MODEL'
	for is_val in [True, False]:
		predictions, real_frames = evaluate_refinenet(P, G_model, is_val)

		if is_val:
			pkl.dump(predictions, open(P['save_dir'] + 'predictions.pkl', 'w'))
			t_str = 'val'
		else:
			t_str = 'train'

		print 'MAKING PLOTS'
		make_evaluation_plots(predictions[:P['n_plot']], real_frames[:P['n_plot']], P['save_dir']+'plots/', 'run_'+str(P['run_num'])+'_'+t_str, t_str+'_')




def train_refinenet(P):

	G_model, D_model = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])

	G_model = initialize_weights(P['G_initial_weights'], G_model)
	D_model = initialize_weights(P['D_initial_weights'], D_model)

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
	print 'Compiling'
	G_model.compile(optimizer=G_opt, loss=P['G_loss'], obj_weights=P['G_obj_weights'])
	D_model.compile(optimizer=D_opt, loss=P['D_loss'])
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

	data = hkl.load(open(P['train_file'], 'r'))
	data_val = hkl.load(open(P['val_file'], 'r'))
	G_data_val = {'mse_frames': data_val['pred'], 'real_frames': data_val['actual']}
	n_ex = data['pred'].shape[0]

	# essentially a place holder
	y = np.zeros((P['batch_size'],2), int)
	y[:,1] = 1

	D_data = {}
	G_data = {}
	D_data['output'] = y
	G_data['output'] = y

	for batch in range(P['num_batches']):

		if batch % P['print_interval']==0:
			verbose=1
		else:
			verbose=0
		if verbose:
			print 'Batch: '+str(batch)
			print '  Discriminator:'

		for d_step in range(P['nb_D_steps']):
			D_idx = np.random.permutation(n_ex)[:P['batch_size']]

			D_data['real_frames'] = data['actual'][D_idx]
			D_data['G_output'] = G_model.predict({'mse_frames': data['pred'][D_idx], 'real_frames': data['actual'][D_idx]})['pixel_output']

			if verbose:
				yhat = D_model.predict(D_data)['output']
				print '   Pre-training- prob fake is best: '+str(yhat[:,1].mean())

			D_model.fit(D_data, batch_size=P['batch_size'], nb_epoch=1, verbose=verbose)

			if verbose:
				yhat = D_model.predict(D_data)['output']
				print '   Post-training- prob fake is best: '+str(yhat[:,1].mean())

		if verbose:
			print 'Batch: '+str(batch)
			print '  Generator:'

		for g_step in range(P['nb_G_steps']):
			if P['same_batches']:
				G_idx = D_idx
			else:
				G_idx = np.random.permutation(n_ex)[:P['batch_size']]

			G_data['real_frames'] = data['actual'][G_idx]
			G_data['mse_frames'] =  data['pred'][G_idx]
			G_data['pixel_output'] = data['actual'][G_idx]

			if verbose:
				D_data['real_frames'] = data['actual'][G_idx]
				D_data['G_output'] = G_model.predict({'mse_frames': data['pred'][G_idx], 'real_frames': data['actual'][G_idx]})['pixel_output']

				#G_feats = extract_features_graph(G_model, ['conv0_D_fake','conv1_D_fake','conv2_D_fake','dense0_D_fake', 'satlu0'], G_data)
				#D_feats = extract_features_graph(D_model, ['conv0_fake','conv1_fake','conv2_fake','dense0_fake'], D_data)

				#yhat_D = D_model.predict(D_data)['output']
				yhat = G_model.predict(G_data)['output']
				print '   Pre-training- prob fake is best: '+str(yhat[:,1].mean())
				#print '   Pre-training- prob fake is best D: '+str(yhat_D[:,1].mean())
				#pdb.set_trace()

			G_model.fit(G_data, batch_size=P['batch_size'], nb_epoch=1, verbose=verbose)

			if verbose:
				yhat = G_model.predict(G_data)['output']
				print '   Post-training- prob fake is best: '+str(yhat[:,1].mean())
				print ''

		if batch % P['val_frequency']==0:
			pred = G_model.predict(G_data_val)
			yhat = pred['output']
			mse = ((G_data_val['real_frames'] - pred['pixel_output'])**2 ).mean()
			print 'mse: '+str(mse)
			log['val_error'].append(yhat[:,1].mean())
			if mse < best_error:
				best_error = mse
				best_error_batch = batch
				G_best_weights = G_model.get_weights()
				D_best_weights = D_model.get_weights()

		if batch % P['plot_frequency'] ==0:
			val_predictions = G_model.predict(G_data_val, batch_size=P['batch_size'])['pixel_output']
			mse = ((G_data_val['real_frames'] - val_predictions)**2 ).mean()
			print 'mse: '+str(mse)
			make_evaluation_plots(val_predictions[:P['n_eval_int_plot']], G_data_val['real_frames'][:P['n_eval_int_plot']], plt_dir, str(mse), 'val_batch'+str(batch)+'_')

		if P['model_save_frequency'] is not None:
			if batch % P['model_save_frequency'] ==0:
				if batch>0:
					G_model.save_weights(P['save_dir']+'G_model_weights_batch'+str(batch)+'.hdf5')
					D_model.save_weights(P['save_dir']+'D_model_weights_batch'+str(batch)+'.hdf5')
					#G_model.save_weights(P['save_dir']+'G_model_best_weights_batch'+str(batch)+'.hdf5')
					#D_model.save_weights(P['save_dir']+'D_model_best_weights_batch'+str(batch)+'.hdf5')

	return G_model, D_model, log, G_best_weights, D_best_weights



def evaluate_refinenet(P, model, is_validation):

	if is_validation:
		f_name = 'val_file'
	else:
		f_name = 'train_file'

	data = hkl.load(open(P[f_name], 'r'))
	data = {'mse_frames': data['pred'][:P['n_evaluate']], 'real_frames': data['actual'][:P['n_evaluate']]}

	predictions = model.predict(data, P['batch_size'])['pixel_output']
	real_frames = data['real_frames']

	return predictions, real_frames


def make_evaluation_plots(predictions, actual_frames, save_dir, title_str, save_str):

	for i in range(predictions.shape[0]):

		plt.subplot(1, 2, 1)
		plt.imshow(actual_frames[i,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.title(title_str + ' clip_'+str(i))
		plt.xlabel('Actual')

		plt.subplot(1, 2, 2)
		plt.imshow(predictions[i,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
		plt.gca().axes.get_xaxis().set_ticks([])
		plt.gca().axes.get_yaxis().set_ticks([])
		plt.title(title_str + ' clip_'+str(i))
		plt.xlabel('Predicted')

		plt.savefig(save_dir+save_str+'clip_'+str(i)+'.jpg')


def create_dataset():

	P = {}

	P['weights_file'] = project_dir + 'facegen_runs_server/run_65/model_best_weights.hdf5'
	P['model_params_file'] = project_dir + 'facegen_runs_server/run_65/params.pkl'

	base_save_dir = '/home/bill/Data/RefineNet_Data/Facegen_Rotations/set_'
	P['set_num'] = 0
	#P['set_num'] = gp.get_next_num(base_save_dir)

	P['save_dir'] = base_save_dir + str(P['set_num']) + '/'

	model = load_prednet_model(P['model_params_file'], P['weights_file'])

	P['orig_files'] = {'_train': '/home/bill/Data/FaceGen_Rotations/clipset4/clipstrain.hkl',
		'_val': '/home/bill/Data/FaceGen_Rotations/clipset4/clipsval.hkl',
		'_test': '/home/bill/Data/FaceGen_Rotations/clipset4/clipstest.hkl'}

	if not os.path.exists(P['save_dir']):
		os.mkdir(P['save_dir'])
	for t in P['orig_files']:
		data = {}
		data['pred'], data['actual'] = get_prednet_predictions(P['model_params_file'], model, P['orig_files'][t], get_data_size(P['orig_files'][t])[0])
		for key in data:
			if data[key].ndim==5:
				s = data[key].shape
				data[key] = data[key].reshape((s[0]*s[1], )+s[2:])
		hkl.dump(data, open(P['save_dir']+'data'+t+'.hkl','w'))

	pkl.dump(P, open(P['save_dir']+'params.pkl', 'w'))



if __name__=='__main__':
	try:
		run_refinenet()
		#create_dataset()

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
