import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model

from copy import deepcopy
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import hickle as hkl
import scipy.io as spio

def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *

cname = get_computer_name()

sys.path.append(get_scripts_dir() +'General_Scripts/')
import general_python_functions as gp
sys.path.append('/home/bill/Libraries/keras/')
from keras.datasets import mnist
from keras.utils import np_utils

base_save_dir = '/home/bill/Projects/Predictive_Networks/mnist_runs/'
MNIST_TRAIN_SZ = 60000

def get_prednet_params(param_overrides=None):

	P = {}
	P['pred_model_name'] = 'mnist_cnn2_predictor0'
	P['initial_weights'] = '/home/bill/Projects/Predictive_Networks/mnist_runs/run_27/model_weights.hdf5'
	P['pred_model_params'] = {'n_lstm': 1024, 'outputs': [], 'n_lstm_layers': 1}
	P['data_dir'] = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/'
	P['frames_in_version'] = 'fixed' #'random'
	P['frames_in_params'] = 10 #[10, 20] if =10, then use frames 0-9 for training
	P['frames_predict_ahead'] = 1
	P['frames_predict'] = range(10, 15)  # index of frames to predict
	P['batch_size'] = 128
	P['block_size'] = 25000
	P['epochs_per_block'] = 100 #3
	P['nb_epoch'] = 100
	P['n_clip_versions_to_use'] = 3 #3
	P['loss_function'] = 'mse'
	P['optimizer'] = 'rmsprop'
	P['n_validate'] = 10000
	P['n_save'] = 100
	P['n_plot'] = 50
	P['save_model'] = True
	P['frame_orig_size'] = 28
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	append_params_per_model(P)

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P

def append_params_per_model(P):

	if P['pred_model_name'] == 'mnist_cnn2_predictor0' or P['pred_model_name'] == 'mnist_cnn2_predictor1':
		P['forward_model_name'] = 'mnist_cnn2'
		# dictionary of forward feature layers to match and corresponding output in pred model
		P['forward_feature_layers'] = None #{'dropout1': ('decoder_output', 8), 'dropout0': ('deconv0_output', 14)}
		P['output_size'] = 26
		P['input_layer'] = 'dropout1'
		P['input_size'] = 8
		if P['pred_model_name'] == 'mnist_cnn2_predictor0':
			P['extra_inputs'] = None
		else:
			P['extra_inputs'] = {'horizontal_input_layer1': ('dropout1', 8, 32), 'horizontal_input_layer0': ('dropout0', 14, 32)}

	return P


def get_deconv_net_params(param_overrides=None):

	P = {}
	P['forward_model_name'] = 'mnist_cnn2'
	P['input_layer'] = 'flatten0'
	P['deconv_model_name'] = 'mnist_cnn2_deconv0'
	P['deconv_model_params'] = {'use_saturation': True}
	P['forward_feature_layers'] = {'dropout1': ('decoder_output', 8), 'dropout0': ('deconv0_output', 14)}
	P['nb_epoch'] = 25
	P['n_train'] = 50000
	P['n_validate'] = 10000
	P['output_size'] = 26
	P['loss_function'] = 'mse'
	P['batch_size'] = 128
	P['save_model'] = True
	P['n_plot'] = 25
	P['n_save'] = 100
	P['optimizer'] = 'rmsprop'
	P['run_num'] = gp.get_next_run_num(base_save_dir)
	P['save_dir'] = base_save_dir + 'run_' + str(P['run_num']) + '/'

	if param_overrides is not None:
		for d in param_overrides:
			P[d] = param_overrides[d]

	return P

def run_deconvnet(param_overrides=None):

	P = get_deconv_net_params(param_overrides)

	(X, _), _ = load_mnist()

	frames_train = X[:P['n_train']]
	frames_test = X[P['n_train']:P['n_train']+P['n_validate']]

	predict_frames = flatten_features(resize_image_stack(frames_train, P['output_size']))

	forward_model = load_model(P['forward_model_name'])

	model = initialize_model(P['deconv_model_name'], P['deconv_model_params'])
	out_dict = {'prediction_output': P['loss_function']}
	if P['forward_feature_layers'] is not None:
		for layer in P['forward_feature_layers']:
			out_dict[P['forward_feature_layers'][layer][0]] = P['loss_function']
	print 'Compiling Model...'
	model.compile(P['optimizer'], out_dict)

	input_features = extract_features_graph(forward_model, P['input_layer'], {'input': frames_train})[P['input_layer']]
	data = {'input': input_features}
	if P['forward_feature_layers'] is not None:
		predict_features = {}
		for layer in P['forward_feature_layers']:
			sz = P['forward_feature_layers'][layer][1]
			predict_features[layer] = extract_features_graph(forward_model, layer, {'input': frames_train})[layer]
			if predict_features[layer].shape[-1] != sz:
				predict_features[layer] = resize_image_stack(predict_features[layer], sz)
			predict_features[layer] = flatten_features(predict_features[layer])

	data = {'input': input_features, 'prediction_output': predict_frames}
	if P['forward_feature_layers'] is not None:
		for layer in P['forward_feature_layers']:
			data[P['forward_feature_layers'][layer][0]] = predict_features[layer]
	model.fit(data, batch_size=P['batch_size'], nb_epoch=P['nb_epoch'], verbose=1)

	input_features = extract_features_graph(forward_model, P['input_layer'], {'input': frames_test})[P['input_layer']]
	data = {'input': input_features}
	predictions = model.predict(data)['prediction_output'].reshape( (P['n_validate'], 1, P['output_size'], P['output_size']) )

	os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	frames_test = resize_image_stack(frames_test, P['output_size'])
	f = open(P['save_dir'] + 'predictions.pkl', 'w')
	pickle.dump([predictions[:P['n_save']], frames_test[:P['n_save']]], f)
	f.close()

	if P['save_model']:
		model.save_weights(P['save_dir']+'model_weights.hdf5')

	pic_save_dir = P['save_dir'] + 'images/'
	title_str = 'Run '+str(P['run_num'])
	tags = []
	for k in range(P['n_plot']):
		tags.append('im'+str(k))
	plot_mnist_prediction_results(frames_test[:P['n_plot']], predictions[:P['n_plot']], pic_save_dir, tags, title_str)


def plot_original_sequences():

	n_plot = 50
	t_steps = range(5, 11)

	actual_frames = load_frames2('/home/bill/Data/MNIST/Rotation_Clips/clip_set1/', 28, t_steps, 25000, 10000, 0, 0, is_validation=True)
	actual_frames = actual_frames[:n_plot]

	plt.figure()
	for i in range(n_plot):
		for j,t in enumerate(t_steps):
			plt.imshow(actual_frames[i,j,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
			plt.title('im_'+str(i)+' t_'+str(t))
			plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/sequence_plots_clipset1_version0_validation/im_'+str(i)+'_t_'+str(t)+'.jpg')


def load_mnist():

	nb_classes = 10

	# the order has already been shuffled
	(X_train, y_train), (X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
	X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
	X_train = X_train.astype("float32")
	X_test = X_test.astype("float32")
	X_train /= 255
	X_test /= 255

	# convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)

	return (X_train, y_train), (X_test, y_test)


def create_rotation_clip(orig_im, center_x, center_y, theta0, angular_speed, n_frames):

	clip = np.zeros((n_frames, orig_im.shape[0], orig_im.shape[1]))
	for i in range(n_frames):
		clip[i] = rotate_image(orig_im, theta0+i*angular_speed, center_x, center_y)

	return clip


def generate_clip_set(orig_ims, n_frames):

	n_ims = orig_ims.shape[0]
	nx = orig_ims.shape[2]
	clip_frames = np.zeros((n_ims, n_frames, 1, 28, 28))

	for i in range(n_ims):
		center_x = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
		center_y = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
		theta0 = 0.0
		angular_speed = np.random.uniform(-np.pi/6, np.pi/6)
		clip_frames[i,:,0] = create_rotation_clip(orig_ims[i,0], center_x, center_y, theta0, angular_speed, n_frames)
		if i%100==0:
			print 'Done clip '+str(i+1)+'/'+str(n_ims)

	return clip_frames


def rotate_image(im, theta, center_x, center_y, threshold = 0.01):

	h = im.shape[0]
	n = np.ceil(h*np.sqrt(2))
	dh = np.ceil((n-h)/2)
	im_large = np.zeros((h+2*n, h+2*n))
	im_large[dh:dh+h,dh:dh+h] = im
	im_rotated = np.zeros((h+2*n, h+2*n))
	R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	for i in range(im.shape[0]):
		dx = i-center_x
		for j in range(im.shape[1]):
			dy = j-center_y
			dv = np.array([[dx], [dy]])
			v = np.dot(R, dv)
			y = center_y+v[1]
			x = center_x+v[0]
			i_low = int(np.floor(x))
			i_high = int(np.ceil(x))
			di = x - i_low
			j_low = int(np.floor(y))
			j_high = int(np.ceil(y))
			dj = y - j_low
			if di<0.5 and dj<0.5:
				v1 = im_large[dh+i_low, dh+j_low]
				v2 = im_large[dh+i_high, dh+j_low]
				v3 = im_large[dh+i_low, dh+j_high]
				val = v1+di*(v2-v1)+dj*(v3-v1)
			elif di<0.5 and dj>=0.5:
				v1 = im_large[dh+i_low, dh+j_high]
				v2 = im_large[dh+i_low, dh+j_low]
				v3 = im_large[dh+i_high, dh+j_high]
				val = v1+di*(v2-v1)+(1-dj)*(v3-v1)
			elif di>=0.5 and dj<0.5:
				v1 = im_large[dh+i_high, dh+j_low]
				v2 = im_large[dh+i_low, dh+j_low]
				v3 = im_large[dh+i_high, dh+j_high]
				val = v1+(1-di)*(v2-v1)+dj*(v3-v1)
			else:
				v1 = im_large[dh+i_high, dh+j_high]
				v2 = im_large[dh+i_low, dh+j_high]
				v3 = im_large[dh+i_high, dh+j_low]
				val = v1+(1-di)*(v2-v1)+(1-dj)*(v3-v1)
			im_rotated[dh+i,dh+j]= val

	im_rotated = im_rotated[dh:dh+h,dh:dh+h]
	im_rotated[im_rotated<threshold] = 0

	return im_rotated


def run_generate_clips():

	n_frames = 20
	base_folder = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/'
	n_clips_per_block = 10000
	clip_versions = [2]

	X = {}
	(X['train'], _), (X['test'], _) = load_mnist()

	for t in ['train','test']:
		for i in clip_versions:
			for j in range(X[t].shape[0]/n_clips_per_block):
				clip_frames = generate_clip_set(X[t][j*n_clips_per_block:(j+1)*n_clips_per_block], n_frames)
				if clip_frames.shape[0]!=n_clips_per_block:
					raise Exception('Clips arent lined up!')
				fname = base_folder + 'mnist_rotations_'+t+'_block'+str(j)+'_clips'+str(i)+'.hkl'
				hkl.dump(clip_frames, fname, mode='w')



def get_block_num_timesteps(P):

	if P['frames_in_version'] == 'random':
		nt = np.random.randint(P['frames_in_params'][0], P['frames_in_params'][1])
	elif P['frames_in_version'] == 'fixed':
		nt = P['frames_in_params']

	return nt

def get_val_idx(P):

	val_idx = []
	for j in range(P['n_clip_versions_to_use']):
		start_i = (j+1)*MNIST_TRAIN_SZ-P['n_validate']
		end_i = start_i + P['n_validate']
		val_idx.extend([i for i in range(start_i, end_i)])

	return val_idx


def get_block_idxs(P):

	n_ex = P['n_clip_versions_to_use']*MNIST_TRAIN_SZ
	all_idx = set([i for i in range(n_ex)])
	if P['n_validate'] != 0:
		all_idx = all_idx - set(get_val_idx(P))

	all_idx = np.random.permutation(list(all_idx))
	idxs = []
	n_blocks = int(np.ceil(float(len(all_idx))/P['block_size']))
	for j in range(n_blocks):
		start_i = j*P['block_size']
		if j==n_blocks:
			end_i = len(all_idx)
		else:
			end_i = (j+1)*P['block_size']
		idxs.append(all_idx[start_i:end_i])
	#pdb.set_trace()

	return idxs


def get_block_per_epoch(P):

	block_nums = np.zeros(P['nb_epoch'], int)
	version_nums = np.zeros(P['nb_epoch'], int)

	all_blocks = []
	for i in range(P['n_clip_versions_to_use']):
		for j in range(int(np.ceil( (MNIST_TRAIN_SZ-P['n_validate'])/P['block_size'] ))):
			all_blocks.append((j,i))

	curr_position = 0
	b_idx = np.random.permutation(range(len(all_blocks)))
	n_repeat = 0
	while curr_position<P['nb_epoch']:
		if n_repeat>=P['epochs_per_block']:
			if len(b_idx)==1:
				b_idx = np.random.permutation(range(len(all_blocks)))
			else:
				b_idx = b_idx[1:]
			n_repeat = 0
		block_nums[curr_position] = all_blocks[b_idx[0]][0]
		version_nums[curr_position] = all_blocks[b_idx[0]][1]
		curr_position += 1
		n_repeat += 1

	return block_nums, version_nums


def get_block_order(P):

	n_blocks = int(np.round(float(P['nb_epoch'])/P['epochs_per_block']))

	all_blocks = []
	for i in range(P['n_clip_versions_to_use']):
		for j in range(int(np.ceil( (MNIST_TRAIN_SZ-P['n_validate'])/P['block_size'] ))):
			all_blocks.append((j,i))

	block_nums = np.zeros(n_blocks, int)
	version_nums = np.zeros(n_blocks, int)

	curr_position = 0
	b_idx = np.random.permutation(range(len(all_blocks)))
	n_repeat = 0
	while curr_position<n_blocks:
		block_nums[curr_position] = all_blocks[b_idx[0]][0]
		version_nums[curr_position] = all_blocks[b_idx[0]][1]
		curr_position += 1
		if len(b_idx)==1:
			b_idx = np.random.permutation(range(len(all_blocks)))
		else:
			b_idx = b_idx[1:]

	return block_nums, version_nums


def test_prednet():

	P = get_prednet_params()
	#
	# forward_model = load_model(P['forward_model_name'])
	#
	# nt = 10
	#
	# input_features = load_features_mnist2(P['data_dir'], P['forward_model_name'], P['input_layer'], P['input_size'], range(nt), P['block_size'], P['n_validate'], 0, 0)
	#
	# predict_features = {}
	# for layer in P['forward_feature_layers']:
	# 	sz = P['forward_feature_layers'][layer][1]
	# 	print '     Loading '+layer+' Features'
	# 	t0 = time.time()
	# 	#predict_features[layer] = load_features_mnist(P['data_dir'], P['forward_model_name'], layer, sz, nt, idx)
	# 	predict_features[layer] = load_features_mnist2(P['data_dir'], P['forward_model_name'], layer, sz, nt, P['block_size'], P['n_validate'], 0, 0)
	#
	# predict_frames = load_frames2(P['data_dir'], P['output_size'], nt, P['block_size'], P['n_validate'], 0, 0)
	#
	# orig_frames = load_frames2(P['data_dir'], P['output_size'], range(nt), P['block_size'], P['n_validate'], 0, 0)
	#
	# plt.figure()
	# for i in range(nt):
	# 	plt.imshow(orig_frames[0,i,0])
	# 	plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/debug/orig_'+str(i)+'.jpg')
	# plt.imshow(predict_frames[0,0])
	# plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/debug/predict.jpg')
	# input2 = extract_features_graph(forward_model, 'dropout1', {'input': orig_frames[0]})
	# plt.imshow(np.mean(input2['dropout1'][0],axis=0))

	#feat = load_features_mnist(P['data_dir'], P['forward_model_name'], 'dropout0', 14, 10, [0,1,2])

	fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/mnist_rotations_train_block'+str(1)+'_clips'+str(2)+'.hkl'
	f = open(fname, 'r')
	these_frames = hkl.load(f)
	f.close()

	pdb.set_trace()


def run_prednet(param_overrides=None):

	P = get_prednet_params(param_overrides)

	forward_model = load_model(P['forward_model_name'])

	model = initialize_model(P['pred_model_name'], P['pred_model_params'])
	out_dict = {'prediction_output': P['loss_function']}
	if P['forward_feature_layers'] is not None:
		for layer in P['forward_feature_layers']:
			out_dict[P['forward_feature_layers'][layer][0]] = P['loss_function']
	print 'Compiling Model...'
	model.compile(P['optimizer'], out_dict)

	if P['initial_weights'] is not None:
		model.load_weights(P['initial_weights'])

	block_nums, version_nums = get_block_order(P)
	for epoch in range(len(block_nums)):

		t_epoch_start = time.time()
		print 'Starting Epoch: '+str(epoch)

		#input_features = load_features_mnist(P['data_dir'], P['forward_model_name'], P['input_layer'], P['input_size'], range(nt), idx)
		nt = get_block_num_timesteps(P)
		print '     Loading Input Features'
		t0 = time.time()
		input_features = load_features_mnist2(P['data_dir'], P['forward_model_name'], P['input_layer'], P['input_size'], range(nt), P['block_size'], P['n_validate'], block_nums[epoch], version_nums[epoch])
		if P['extra_inputs'] is None:
			extra_inputs = None
		else:
			extra_inputs = {}
			for inp in P['extra_inputs']:
				extra_inputs[inp] = load_features_mnist2(P['data_dir'], P['forward_model_name'], P['extra_inputs'][inp][0], P['extra_inputs'][inp][1], nt-1, P['block_size'], P['n_validate'], block_nums[epoch], version_nums[epoch])
				extra_inputs[inp] = np.reshape(extra_inputs[inp], (extra_inputs[inp].shape[0], P['extra_inputs'][inp][2], P['extra_inputs'][inp][1], P['extra_inputs'][inp][1]))
		print '         time elapsed: '+str(np.round(time.time()-t0)) +' seconds'

		predict_features = {}
		if P['forward_feature_layers'] is not None:
			for layer in P['forward_feature_layers']:
				sz = P['forward_feature_layers'][layer][1]
				print '     Loading '+layer+' Features'
				t0 = time.time()
				#predict_features[layer] = load_features_mnist(P['data_dir'], P['forward_model_name'], layer, sz, nt, idx)
				predict_features[layer] = load_features_mnist2(P['data_dir'], P['forward_model_name'], layer, sz, nt, P['block_size'], P['n_validate'], block_nums[epoch], version_nums[epoch])
				print '         time elapsed: '+str(np.round(time.time()-t0)) +' seconds'

		predict_frames = flatten_features(load_frames2(P['data_dir'], P['output_size'], nt, P['block_size'], P['n_validate'], block_nums[epoch], version_nums[epoch]))
		data = {'input': input_features, 'prediction_output': predict_frames}

		if P['forward_feature_layers'] is not None:
			for layer in P['forward_feature_layers']:
				data[P['forward_feature_layers'][layer][0]] = predict_features[layer]
		if extra_inputs is not None:
			for inp in extra_inputs:
				data[inp] = extra_inputs[inp]
		print '     Fitting Batch'
		t0 = time.time()
		model.fit(data, batch_size=P['batch_size'], nb_epoch=P['epochs_per_block'], verbose=1)
		del input_features, predict_features, predict_frames, data
		print '         time elapsed: '+str(np.round(time.time()-t0)) +' seconds'
		print ' time for epoch: '+str(np.round(time.time()-t_epoch_start))

	os.mkdir(P['save_dir'])

	f = open(P['save_dir'] + 'params.pkl', 'w')
	pickle.dump(P, f)
	f.close()

	if P['save_model']:
		model.save_weights(P['save_dir']+'model_weights.hdf5')

	predictions, actual_frames = evaluate_prednet(P, forward_model, model)

	f = open(P['save_dir'] + 'predictions.pkl', 'w')
	pickle.dump([predictions[:P['n_save']], actual_frames[:P['n_save']]], f)
	f.close()

	if P['n_plot'] != 0:
		print 'Making Plots'
		pic_save_dir = P['save_dir'] + 'images/'
		title_str = 'Run '+str(P['run_num'])
		for i,t in enumerate(P['frames_predict']):
			tags = []
			for k in range(P['n_plot']):
				tags.append('im'+str(k)+'_t'+str(t))
			plot_mnist_prediction_results(actual_frames[:P['n_plot'],i], predictions[:P['n_plot'],i], pic_save_dir, tags, title_str)


def plot_run(run_num):

	run_dir = base_save_dir + 'run_'+str(run_num)+'/'
	f = open(run_dir + 'params.pkl', 'r')
	P = pickle.load(f)
	f.close()

	f = open(run_dir + 'predictions.pkl', 'r')
	predictions, actual_frames = pickle.load(f)
	f.close()

	pic_save_dir = P['save_dir'] + 'images/'
	title_str = 'Run '+str(P['run_num'])
	for i,t in enumerate(P['frames_predict']):
		tags = []
		for k in range(P['n_plot']):
			tags.append('im'+str(k)+'_t'+str(t))
		plot_mnist_prediction_results(actual_frames[:P['n_plot'],i], predictions[:P['n_plot'],i], pic_save_dir, tags, title_str)

def load_params(run_num):

	run_dir = base_save_dir + 'run_'+str(run_num)+'/'
	f = open(run_dir + 'params.pkl', 'r')
	P = pickle.load(f)
	f.close()

	return P

def load_predictions(run_num):

	run_dir = base_save_dir + 'run_'+str(run_num)+'/'
	f = open(run_dir + 'predictions.pkl', 'r')
	predictions, actual_frames = pickle.load(f)
	f.close()

	return predictions, actual_frames


def finish_run(run_num):

	run_dir = base_save_dir + 'run_'+str(run_num)+'/'
	f = open(run_dir + 'params.pkl', 'r')
	P = pickle.load(f)
	f.close()

	model = initialize_model(P['pred_model_name'], P['pred_model_params'])
	model.load_weights(run_dir + 'model_weights.hdf5')
	out_dict = {'prediction_output': P['loss_function']}
	for layer in P['forward_feature_layers']:
		out_dict[P['forward_feature_layers'][layer][0]] = P['loss_function']
	print 'Compiling Model...'
	model.compile(P['optimizer'], out_dict)

	forward_model = load_model(P['forward_model_name'])

	predictions, actual_frames = evaluate_prednet(P, forward_model, model)

	f = open(P['save_dir'] + 'predictions.pkl', 'w')
	pickle.dump([predictions[:P['n_save']], actual_frames[:P['n_save']]], f)
	f.close()

	plot_run(run_num)


def evaluate_prednet(P, forward_model, model):

	#true_features = load_features_mnist(P['data_dir'], P['forward_model_name'], P['input_layer'], P['input_size'], range(P['frames_predict'][0]), val_idx)
	true_features = load_features_mnist2(P['data_dir'], P['forward_model_name'], P['input_layer'], P['input_size'], range(P['frames_predict'][0]), P['block_size'], P['n_validate'], 0, 0, is_validation=True)
	pred_features = {}
	pred_features['input'] = np.zeros((P['n_validate'], P['frames_predict'][-1]+1, true_features.shape[-1]))
	pred_features['input'][:,:P['frames_predict'][0]] = true_features
	if P['extra_inputs'] is not None:
		for inp in P['extra_inputs']:
			pred_features[inp] = load_features_mnist2(P['data_dir'], P['forward_model_name'], P['extra_inputs'][inp][0], P['extra_inputs'][inp][1], P['frames_predict'][0]-1, P['block_size'], P['n_validate'], 0, 0, is_validation=True)
			pred_features[inp] = np.reshape(pred_features[inp], (pred_features[inp].shape[0], P['extra_inputs'][inp][2], P['extra_inputs'][inp][1], P['extra_inputs'][inp][1]))
	predictions = np.zeros((P['n_validate'], len(P['frames_predict']), 1, P['output_size'], P['output_size']))
	del true_features
	for i,t in enumerate(P['frames_predict']):
		data = {'input': pred_features['input'][:,:t]}
		if P['extra_inputs'] is not None:
			for inp in P['extra_inputs']:
				data[inp] = pred_features[inp]
		y_hat = model.predict(data)['prediction_output'].reshape( (P['n_validate'], 1, P['output_size'], P['output_size']) )
		predictions[:,i] = y_hat
		# append to pred_featur
		if P['output_size'] != P['frame_orig_size']:
			next_frames = resize_image_stack(y_hat, P['frame_orig_size'])
		else:
			next_frames = y_hat
		pred_features['input'][:,t] = flatten_features(extract_features_graph(forward_model, P['input_layer'], {'input': next_frames})[P['input_layer']])
		if P['extra_inputs'] is not None:
			for inp in P['extra_inputs']:
				feat = extract_features_graph(forward_model, P['extra_inputs'][inp][0], {'input': next_frames})[P['extra_inputs'][inp][0]]
				if feat.shape[-1] != P['extra_inputs'][inp][1]:
					feat = resize_image_stack(feat, P['extra_inputs'][inp][1])
				pred_features[inp] = feat
	#actual_frames = load_frames(P['data_dir'], P['output_size'], P['frames_predict'], val_idx)
	actual_frames = load_frames2(P['data_dir'], P['output_size'], P['frames_predict'], P['block_size'], P['n_validate'], 0, 0, is_validation=True)

	return predictions, actual_frames


def load_features_mnist2(base_dir, model_name, layer, size, time_steps, block_size, n_validate, block_num, version_num, is_validation=False):

	if isinstance(time_steps, int):
		time_steps = [time_steps]

	feat_dir = base_dir + 'model_features/'+model_name+'/'+layer+'/size_'+str(size)+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
	for i,t in enumerate(time_steps):
		fname = feat_dir + 'features_t'+str(t)+'_block'+str(block_num)+'_version'+str(version_num)
		if is_validation:
			fname += '_val'
		fname += '.hkl'
		f = open(fname, 'r')
		these_features = hkl.load(f)
		f.close()

		if len(time_steps)==1:
			features = these_features
		else:
			if i==0:
				if is_validation:
					n_ex = n_validate
				else:
					n_ex = block_size
				features = np.zeros((n_ex, len(time_steps), these_features.shape[-1]))
			features[:,i] = these_features

	return features


def load_frames2(base_dir, size, time_steps, block_size, n_validate, block_num, version_num, is_validation = False):

	if isinstance(time_steps, int):
		time_steps = [time_steps]

	feat_dir = base_dir + 'single_frames/size_'+str(size)+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
	for i,t in enumerate(time_steps):
		fname = feat_dir + 'frame_t'+str(t)+'_block'+str(block_num)+'_version'+str(version_num)
		if is_validation:
			fname += '_val'
		fname += '.hkl'
		f = open(fname, 'r')
		these_features = hkl.load(f)
		f.close()

		if len(time_steps)==1:
			features = these_features
		else:
			if i==0:
				if is_validation:
					n_ex = n_validate
				else:
					n_ex = block_size
				features = np.zeros((n_ex, len(time_steps), 1, size, size))
			features[:,i] = these_features

	return features


def load_features_mnist(base_dir, model_name, layer, size, time_steps, idx):

	if not isinstance(time_steps, list):
		time_steps = [time_steps]

	feat_dir = base_dir + 'model_features/'+model_name+'/'+layer+'/size_'+str(size)+'/'
	idx = np.array(idx)
	for t_idx, t in enumerate(time_steps):

		if layer=='dropout1':
			fname = feat_dir + 'features_'+str(t)+'.hkl'
			f = open(fname, 'r')
			these_features = hkl.load(f)[idx]
			f.close()
		elif layer=='dropout0':
			initialized = False
			for i in range(9):
				in_batch = np.nonzero(np.logical_and(idx>=i*20000, idx<(i+1)*20000))[0]
				if len(in_batch)>0:
					fname = feat_dir + 'features_'+str(t)+'_20000_'+str(i)+'.hkl'
					f = open(fname, 'r')
					these_f = hkl.load(f)
					f.close()
					this_idx = idx[in_batch] - i*20000
					these_f = these_f[this_idx]
					if not initialized:
						these_features = np.zeros((len(idx), these_f.shape[-1])).astype(np.float32)
						initialized = True
					these_features[in_batch] = these_f

		if len(time_steps)==1:
			features = these_features
		else:
			if t_idx==0:
				features = np.zeros((len(idx), len(time_steps), these_features.shape[-1]))
			features[:,t_idx] = these_features

	return features


def load_frames(base_dir, frame_size, time_steps, idx):

	if not isinstance(time_steps, list):
		time_steps = [time_steps]

	frame_dir = base_dir + 'single_frames/size_'+str(frame_size)+'/'
	for i,t in enumerate(time_steps):
		fname = frame_dir + 'frame_'+str(t)+'.hkl'
		f = open(fname, 'r')
		these_frames = hkl.load(f)[idx]
		f.close()
		if len(time_steps)==1:
			return these_frames
		else:
			if i==0:
				s = these_frames.shape
				frames = np.zeros((len(idx), len(time_steps), s[1], s[2], s[3]))
			frames[:,i] = these_frames

	return frames


def load_test():

	fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/mnist_rotations_train_block0_clips1.hkl'
	f = open(fname, 'r')

	t0 = time.time()
	vals = hkl.load(f)
	print 'Time to load 10k using hkl: '+str(time.time()-t0)
	f.close()

	fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/mnist_rotations_train_block0_clips3.hkl'
	f = open(fname, 'r')

	t0 = time.time()
	vals2 = hkl.load(f)
	print 'Time to load 10k using hkl: '+str(time.time()-t0)
	f.close()

	pdb.set_trace()

def split_data():

	# for each of features and frames, split into chunks for easy training
	block_size = 25000
	n_validate = 10000
	validation = True
	model_name = 'mnist_cnn2'
	# block size, n_validate and num clip versions dictate sets
	# will be like features_[time_step]_block0_version
	#layers = {'dropout0': 14, 'dropout1': 8}
	layers = {'dropout1': 8}
	frame_sizes = [26]
	#layers = {}
	#frame_sizes = []
	#layers = {'dropout0': 14}
	if validation:
		b_range = 1
	else:
		b_range = 2
	if len(layers)>0:
		for l in layers:
			out_dir = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/model_features/'+model_name+'/'+l+'/size_'+str(layers[l])+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			for t in range(20):
				for v in range(3):
					for b in range(b_range):
						if validation:
							idx = [i+v*MNIST_TRAIN_SZ+50000 for i in range(n_validate)]
							fname = out_dir + 'features_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'_val.hkl'
						else:
							idx = [i+v*MNIST_TRAIN_SZ+b*block_size for i in range(block_size)]
							fname = out_dir + 'features_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'.hkl'
						features = load_features_mnist('/home/bill/Data/MNIST/Rotation_Clips/clip_set1/', model_name, l, layers[l], t, idx)

						print fname
						f = open(fname, 'w')
						hkl.dump(features, f)

	if len(frame_sizes)>0:
		for s in frame_sizes:
			out_dir = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/single_frames/size_'+str(s)+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			for t in range(20):
				for v in range(3):
					for b in range(b_range):
						if validation:
							idx = [i+v*MNIST_TRAIN_SZ+50000 for i in range(n_validate)]
							fname = out_dir + 'frames_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'_val.hkl'
						else:
							idx = [i+v*MNIST_TRAIN_SZ+b*block_size for i in range(block_size)]
							fname = out_dir + 'frames_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'.hkl'
						frames = load_frames('/home/bill/Data/MNIST/Rotation_Clips/clip_set1/', s, t, idx)

						print fname
						f = open(fname, 'w')
						hkl.dump(frames, f)

def save_single_frames():

	n_versions = 3
	output_size = 28
	n_blocks = 6

	out_folder = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/single_frames/size_'+str(output_size)+'/'
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)

	for i in range(13,20):
		print 'Starting frame '+str(i)
		counter = 0
		all_frames = np.zeros((60000*n_versions, 1, output_size, output_size))
		for j in range(n_versions):
			for k in range(n_blocks):
				fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/mnist_rotations_train_block'+str(k)+'_clips'+str(j)+'.hkl'
				f = open(fname, 'r')
				these_frames = hkl.load(f)
				f.close()
				s = np.shape(these_frames)
				these_frames = np.reshape(these_frames[:,i], (s[0], s[2], s[3], s[4]))
				if these_frames.shape[-1] != output_size:
					all_frames[counter:counter+s[0]] = resize_image_stack(these_frames, output_size)
				counter += s[0]
				print counter
		fname = out_folder+'frame_'+str(i)+'.hkl'
		f = open(fname, 'w')
		hkl.dump(all_frames, f)

def flatten_features(X):
	size = np.prod(X.shape)/X.shape[0]
	return X.reshape( (X.shape[0], size) )

def save_model_features():

	# features are calculated from original image size
	model_name = 'mnist_cnn2'
	#layers = ['dropout1']
	#sizes = [8]
	#t_steps = range(10)
	layers = ['dropout0']
	sizes = [14]
	t_steps = range(10, 20)
	#layers = ['flatten0']
	#sizes = [None]
	#t_steps = range(20)
	d_dir = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/'
	frames_dir = d_dir+'single_frames/size_28/'
	batch_size = 20000

	model = load_model(model_name)

	for i,l in enumerate(layers):
		for t in t_steps:
			print 'Layer '+l+' time '+str(t)
			fname = frames_dir+'frame_'+str(t)+'.hkl'
			f = open(fname, 'r')
			frames = hkl.load(f)
			f.close()
			# frames will be (n_clips, 1, 28, 28)
			for j in range(frames.shape[0]/batch_size):
				features = extract_features_graph(model, l, {'input': frames[j*batch_size:(j+1)*batch_size]})[l]
				if sizes[i] is not None:
					if sizes[i] != features.shape[-1]:
						features = resize_image_stack(features, sizes[i])
				features = flatten_features(features)
				print features.shape
				out_dir = d_dir+'model_features/'+model_name+'/'+l+'/size_'+str(sizes[i])+'/'
				if not os.path.exists(out_dir):
					os.makedirs(out_dir)
				fname = out_dir+'features_'+str(t)+'_'+str(batch_size)+'_'+str(j)+'.hkl'
				f = open(fname, 'w')
				hkl.dump(features, f)
				f.close()


def save_model_features2():

	layer = 'dropout0'
	size = 14
	model_name = 'mnist_cnn2'
	d_dir = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/'
	frames_dir = d_dir+'single_frames/size_28/'
	block_size = 25000
	n_validate = 10000
	t_steps = [9] #range(20)
	n_versions = 3
	validation = True

	if validation:
		b_range = 1
	else:
		b_range = 2

	model = load_model(model_name)

	out_dir = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/model_features/'+model_name+'/'+layer+'/size_'+str(size)+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
	frames_dir += 'blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for t in t_steps:
		for v in range(n_versions):
			for b in range(b_range):
				if validation:
					frames_fname = frames_dir+'frame_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'_val.hkl'
					fname = out_dir + 'features_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'_val.hkl'
				else:
					frames_fname = frames_dir+'frame_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'.hkl'
					fname = out_dir + 'features_t'+str(t)+'_block'+str(b)+'_version'+str(v)+'.hkl'

				f = open(frames_fname, 'r')
				frames = hkl.load(f)
				f.close()

				features = extract_features_graph(model, layer, {'input': frames})[layer]
				if size != features.shape[-1]:
					features = resize_image_stack(features, size)
				features = flatten_features(features)

				f = open(fname, 'w')
				hkl.dump(features, f)
				f.close()


def save_single_frames2():

	t_steps = range(4, 20)
	versions = [2]
	output_size = 28
	block_size = 25000
	n_validate = 10000

	out_folder = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/single_frames/size_'+str(output_size)+'/blocksize_'+str(block_size)+'_nval_'+str(n_validate)+'/'
	if not os.path.exists(out_folder):
		os.mkdir(out_folder)


	for t in t_steps:
		print 'Time '+str(t)
		for v in versions:
			print 'Version '+str(v)
			all_frames = np.zeros((MNIST_TRAIN_SZ, 1, output_size, output_size))
			for k in range(6):
				fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/mnist_rotations_train_block'+str(k)+'_clips'+str(v)+'.hkl'
				f = open(fname, 'r')
				these_frames = hkl.load(f)
				f.close()
				s = np.shape(these_frames)
				these_frames = np.reshape(these_frames[:,t], (s[0], s[2], s[3], s[4]))
				if these_frames.shape[-1] != output_size:
					these_frames = resize_image_stack(these_frames, output_size)
				all_frames[k*10000:(k+1)*10000] = these_frames

			fname = out_folder + 'frame_t'+str(t)+'_block0_version'+str(v)+'.hkl'
			f = open(fname, 'w')
			hkl.dump(all_frames[:block_size], f)
			f.close()
			fname = out_folder + 'frame_t'+str(t)+'_block1_version'+str(v)+'.hkl'
			f = open(fname, 'w')
			hkl.dump(all_frames[block_size:2*block_size], f)
			f.close()
			fname = out_folder + 'frame_t'+str(t)+'_block0_version'+str(v)+'_val.hkl'
			f = open(fname, 'w')
			hkl.dump(all_frames[2*block_size:], f)
			f.close()






# X looks like {'input': X_train}
def extract_features_graph(model, layer_names, X):

	#model2 = deepcopy(model)
	model2 = model

	if not isinstance(layer_names, list):
		layer_names = [layer_names]

	for i,l_name in enumerate(layer_names):
		n = 'out'+str(i)
		model2.add_output(name=n, input=l_name)

	ins = [model2.inputs[name].input for name in model2.input_order]
	ys_test = []
	for output_name in model2.output_order:
		output = model2.outputs[output_name]
		y_test = output.get_output(False)
		ys_test.append(y_test)
	model2._predict = theano.function(inputs=ins, outputs=ys_test, allow_input_downcast=True)
	outs = model2.predict(X)
	outputs = {}
	for i,l_name in enumerate(layer_names):
		outputs[l_name] = outs['out'+str(i)]

	return outputs

def resize_image_stack(im_stack, size, threshold = 0.01):

	imstack_resampled = np.zeros((im_stack.shape[0], im_stack.shape[1], size, size)).astype(np.float32)
	for i in range(im_stack.shape[0]):
		for j in range(im_stack.shape[1]):
			im = scipy.ndimage.zoom(im_stack[i,j], float(size)/im_stack.shape[2])
			im[im<threshold] = 0
			imstack_resampled[i,j]= im

	return imstack_resampled


def plot_mnist_prediction_results(y, y_hat, save_dir, tags, title_str):

	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	plt.figure()
	for i in range(y.shape[0]):
		plt.subplot(1, 2, 1)
		plt.imshow(y[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Original')
		plt.title(title_str)

		plt.subplot(1, 2, 2)
		plt.imshow(y_hat[i,0], cmap="Greys", vmin=0.0, vmax=1.0, interpolation='none')
		plt.xlabel('Reconstructed')
		plt.title(tags[i])

		plt.savefig(save_dir + tags[i] + '.jpg')


def create_matlab_frames():

	X = np.zeros((50, 10, 28, 28))

	for i in range(10):

		fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/single_frames/size_28/blocksize_25000_nval_10000/frame_t'+str(i)+'_block0_version0_val.hkl'
		f = open(fname, 'r')
		frames = hkl.load(fname)
		f.close()

		X[:,i] = frames[:50,0]

	out_name = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/MNIST_rotation_val_clips.mat'
	spio.savemat(out_name, {'frames': X})

def create_matlab_predictions():

	f = open('/home/bill/Projects/Predictive_Networks/mnist_runs/run_27/predictions.pkl','r')
	predictions, actual = pickle.load(f)

	predictions = predictions[:50]
	actual = actual[:50]

	# actual = np.zeros((50, 5, 26, 26))
	#
	# for idx,i in enumerate(range(10,15)):
	# 	fname = '/home/bill/Data/MNIST/Rotation_Clips/clip_set1/single_frames/size_26/blocksize_25000_nval_10000/frame_t'+str(i)+'_block0_version0_val.hkl'
	# 	f = open(fname, 'r')
	# 	frames = hkl.load(fname)
	# 	f.close()
	# 	actual[:,idx] = frames[:50,0]

	pdb.set_trace()
	out_name = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/MNIST_rotation_run27_predictions.mat'
	spio.savemat(out_name, {'predictions': predictions, 'actual': actual})

def copy_sequences_to_matlab():

	f = open('/home/bill/Data/MNIST/Shape_Pattern_Clips/clipset_0/val_clips.hkl','r')
	X = hkl.load(f)
	f.close()
	X = X[:30]

	spio.savemat('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/val_sequences.mat', {'frames': X})

def copy_GAN_sequences():

	for r in [6,17]:

		f = open('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/mnist_GAN_runs/run_'+str(r)+'/predictions.pkl', 'r')
		X = pickle.load(f)
		X = X[0][:30]

		spio.savemat('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_MNIST_shift_sequences/run'+str(r)+'_predictions.mat', {'predictions': X})


if __name__=='__main__':
	try:
		#run_generate_clips()
		#load_test()
		#run_prednet()
		#run_prednet({'pred_model_params': {'n_lstm': 2048, 'outputs': [], 'n_lstm_layers': 1}})
		#run_prednet({'pred_model_params': {'n_lstm': 1024, 'outputs': [], 'n_lstm_layers': 2}})
		#finish_run(11)
		#plot_original_sequences()

		#run_deconvnet()
		#run_deconvnet({'loss_function': 'mae'})
		#run_deconvnet({'forward_feature_layers': None})
		#run_deconvnet({'loss_function': 'mae', 'forward_feature_layers': None})
		#split_data()
		#test_prednet()
		#save_model_features2()
		#save_single_frames2()

		#create_matlab_frames()
		#copy_sequences_to_matlab()
		copy_GAN_sequences()
	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
