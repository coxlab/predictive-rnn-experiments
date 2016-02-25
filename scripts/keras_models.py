import sys, traceback
sys.path.append('/home/bill/Libraries/keras/')
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.layers.normalization import *
import pdb
import networkx as nx
import matplotlib.pyplot as plt



def initialize_model(model_name, params = None):

	if model_name == 'mnist_cnn0':
		model = Sequential()

		model.add(Convolution2D(32, 1, 3, 3, border_mode='full'))
		model.add(Activation('relu'))
		model.add(Convolution2D(32, 32, 3, 3))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(poolsize=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(32*196, 128))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(128, 10))
		model.add(Activation('softmax'))

	elif model_name == 'mnist_cnn1':
		model = Sequential()

		model.add(Convolution2D(32, 1, 3, 3))
		model.add(Activation('relu'))
		model.add(Convolution2D(16, 32, 3, 3))
		model.add(Activation('relu'))
		model.add(Dropout(0.25))
		model.add(Convolution2D(16, 16, 3, 3, subsample = (2,2)))
		model.add(Activation('relu'))
		model.add(Dropout(0.25))

		model.add(Flatten())
		model.add(Dense(16*121, 512))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(512, 10))
		model.add(Activation('softmax'))

	elif 'mnist_cnn_deconv0' in model_name:
		# should be like mnist_cnn_deconv0_[n_layers]
		n_layers = int(model_name[-1])

		model = Sequential()

		for i in range(n_layers):
			j = n_layers-1 - i

			if j==0:
				model.add(Convolution2D(1, 32, 3, 3))
				model.add(Activation('relu'))

			elif j==1:
				model.add(Convolution2D(32, 32, 3, 3))
				model.add(Activation('relu'))

			elif j==2:
				model.add(UnPooling2D(unpoolsize=(2, 2)))
				model.add(Dropout(0.25))

			elif j==3:
				model.add(Dense(128, 32*196))
				model.add(Activation('relu'))
				model.add(Dropout(0.5))
				model.add(Reshape(32, 14, 14))

			elif j==4:
				model.add(Dense(10, 128))
				model.add(Activation('relu'))

	elif 'mnist_rotation_predictor0' in model_name:

		n_layers = int(model_name[-1])

		model = Sequential()

		if n_layers==2:

			# input needs to be (nb_samples, timesteps, input_dim)
			#model.add(LSTM(28*28*32, 1024, return_sequences=False))
			model.add(LSTM(1000, 2048, return_sequences=False))
			#model.add(LSTM(28*28*32, 2048, return_sequences=False))
			#model.add(LSTM(1024, 2048, return_sequences=False))
			# output will be (nb_samples, timesteps, output_dim)
			#model.add(CollapseTimesteps(2))
			model.add(Reshape(32, 8, 8))
			model.add(UnPooling2D((2,2)))
			# want to reshape to (nb_samples*timesteps, 32, 8, 8)
			# input needs to be (nb_samples, stack_size,nb_row, nb_col)
			model.add(Convolution2D(32, 32, 3, 3))
			model.add(Activation('relu'))
			model.add(Convolution2D(1, 32, 3, 3))
			model.add(Activation('relu'))
		elif n_layers==4:
			model.add(LSTM(128, 1024, return_sequences=False))
			model.add(Dense(1024, 32*196))
			model.add(Activation('relu'))
			model.add(Dropout(0.5))
			model.add(Reshape(32, 14, 14))
			model.add(UnPooling2D(unpoolsize=(2, 2)))
			model.add(Dropout(0.25))
			model.add(Convolution2D(32, 32, 3, 3))
			model.add(Activation('relu'))
			model.add(Convolution2D(1, 32, 3, 3))
			model.add(Activation('relu'))

	elif model_name == 'mnist_rotation_predandclass0':

		model = Graph()

		model.add_input(name='input', ndim=4)

		model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0', input='input')
		model.add_node(Activation('relu'), name='relu0', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='relu0')
		model.add_node(Dropout(0.25), name='dropout0', input='pool0')

		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full'), name='conv1', input='dropout0')
		model.add_node(Activation('relu'), name='relu1', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='relu1')
		model.add_node(Dropout(0.25), name='dropout1', input='pool1')

		# flatten stacks
		model.add_node(Flatten(), name='flatten0', input='dropout1')

		# classification loss
		model.add_node(Dense(32*7*7, 128), name='classification_dense0', input='flatten0')
		model.add_node(Activation('relu'), name='classification_relu0', input='classification_dense0')
		model.add_node(Dropout(0.5), name='classification_dropout0', input='classification_relu0')

		model.add_node(Dense(128, 10), name='classification_dense1', input='classification_dropout0')
		model.add_node(Activation('softmax'), name='classification_softmax0', input='classification_dense1')
		model.add_output(name='classification_output', input='classification_softmax0')

		#LSTM for prediction
		# output of forward net will be (n_clips*n_frames, shp)
		model.add_node(ExpandTimesteps(params['n_timesteps']), name='LSTM_expansion0', input='flatten0')
		model.add_node(LSTM(32*8*8, 1024, return_sequences=False), name='LSTM0', input='LSTM_expansion0')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		# decoder
		model.add_node(Dense(1024, 32*8*8), name='decoder_dense0', input='LSTM_dropout0')
		model.add_node(Activation('relu'), name='decoder_relu0', input='decoder_dense0')
		model.add_output(name='decoder_output', input='decoder_relu0')
		model.add_node(Reshape(32, 7, 7), name='unflatten0', input='decoder_relu0')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten0')
		model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')

		model.add_node(Flatten(), name='flatten1', input='deconv_dropout0')
		model.add_output(name='deconv0_output', input='flatten1')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Dropout(0.25), name='deconv_dropout1', input='deconv_relu1')

		model.add_node(Flatten(), name='flatten2', input='deconv_dropout1')
		model.add_output(name='prediction_output', input='flatten2')


	elif model_name == 'mnist_cnn2':

		model = Graph()

		model.add_input(name='input', ndim=4)

		model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0', input='input')
		model.add_node(Activation('relu'), name='relu0', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='relu0')
		model.add_node(Dropout(0.25), name='dropout0', input='pool0')

		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full'), name='conv1', input='dropout0')
		model.add_node(Activation('relu'), name='relu1', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='relu1')
		model.add_node(Dropout(0.25), name='dropout1', input='pool1')

		# flatten stacks
		model.add_node(Flatten(), name='flatten0', input='dropout1')

		# classification loss
		model.add_node(Dense(32*8*8, 128), name='classification_dense0', input='flatten0')
		model.add_node(Activation('relu'), name='classification_relu0', input='classification_dense0')
		model.add_node(Dropout(0.5), name='classification_dropout0', input='classification_relu0')

		model.add_node(Dense(128, 10), name='classification_dense1', input='classification_dropout0')
		model.add_node(Activation('softmax'), name='classification_softmax0', input='classification_dense1')
		model.add_output(name='classification_output', input='classification_softmax0')

	elif model_name == 'mnist_cnn2_predictor0':

		if params is None:
			params = {'n_lstm': 1024}
			params['outputs'] = []
			params['n_lstm_layers'] = 1

		model = Graph()

		model.add_input(name='input', ndim=3)

		if params['n_lstm_layers'] > 1:
			rs = True
		else:
			rs = False

		model.add_node(LSTM(32*8*8, params['n_lstm'], return_sequences=rs), name='LSTM0', input='input')
		#model.add_node(LSTM(32*8*8, 32*8*8, return_sequences=False), name='LSTM0', input='input')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		if params['n_lstm_layers'] == 2:
			model.add_node(LSTM(params['n_lstm'], params['n_lstm'], return_sequences=False), name='LSTM1', input='LSTM_dropout0')
			model.add_node(Dropout(0.25), name='LSTM_dropout1', input='LSTM1')

		# decoder
		#model.add_node(Dense(32*8*8, 32*8*8), name='decoder_dense0', input='LSTM_dropout0')
		model.add_node(Dense(params['n_lstm'], 32*8*8), name='decoder_dense0', input='LSTM_dropout'+str(params['n_lstm_layers']-1))
		model.add_node(Activation('relu'), name='decoder_relu0', input='decoder_dense0')
		if 'decoder_output' in params:
			model.add_output(name='decoder_output', input='decoder_relu0')
		model.add_node(Reshape(32, 8, 8), name='unflatten0', input='decoder_relu0')
		#model.add_node(Reshape(32, 8, 8), name='unflatten0', input='LSTM_dropout0')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten0')
		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')

		model.add_node(Flatten(), name='flatten1', input='deconv_dropout0')

		if 'deconv0_output' in params:
			model.add_output(name='deconv0_output', input='flatten1')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='valid'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		#model.add_node(Dropout(0.25), name='deconv_dropout1', input='deconv_relu1')
		model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')

		model.add_node(Flatten(), name='flatten2', input='deconv_satlu')
		model.add_output(name='prediction_output', input='flatten2')


	elif model_name == 'mnist_cnn2_predictor1':

		if params is None:
			n_lstm = 1024
		else:
			n_lstm = params

		model = Graph()

		model.add_input(name='input', ndim=3)

		model.add_node(LSTM(32*8*8, n_lstm, return_sequences=False), name='LSTM0', input='input')
		#model.add_node(LSTM(32*8*8, 32*8*8, return_sequences=False), name='LSTM0', input='input')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		# decoder
		#model.add_node(Dense(32*8*8, 32*8*8), name='decoder_dense0', input='LSTM_dropout0')
		model.add_node(Dense(n_lstm, 32*8*8), name='decoder_dense0', input='LSTM_dropout0')
		model.add_node(Activation('relu'), name='decoder_relu0', input='decoder_dense0')
		model.add_output(name='decoder_output', input='decoder_relu0')
		model.add_node(Reshape(32, 8, 8), name='unflatten0', input='decoder_relu0')
		#model.add_node(Reshape(32, 8, 8), name='unflatten0', input='LSTM_dropout0')

		# deconv net with horizontal connections
		model.add_input(name='horizontal_input_layer1', ndim=4)
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', inputs=['unflatten0', 'horizontal_input_layer1'], concat_axis=1)
		model.add_node(Convolution2D(32, 64, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')

		model.add_node(Flatten(), name='flatten1', input='deconv_dropout0')
		model.add_output(name='deconv0_output', input='flatten1')

		model.add_input(name='horizontal_input_layer0', ndim=4)
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', inputs=['deconv_dropout0', 'horizontal_input_layer0'], concat_axis=1)
		model.add_node(Convolution2D(1, 64, 3, 3, border_mode='valid'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Dropout(0.25), name='deconv_dropout1', input='deconv_relu1')

		model.add_node(Flatten(), name='flatten2', input='deconv_dropout1')
		model.add_output(name='prediction_output', input='flatten2')


	elif model_name == 'mnist_cnn2_deconv0':

		model = Graph()

		model.add_input(name='input', ndim=2)

		model.add_node(Dense(32*8*8, 32*8*8), name='decoder_dense0', input='input')
		model.add_node(Activation('relu'), name='decoder_relu0', input='decoder_dense0')
		model.add_output(name='decoder_output', input='decoder_relu0')
		model.add_node(Reshape(32, 8, 8), name='unflatten0', input='decoder_relu0')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten0')
		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')

		model.add_node(Flatten(), name='flatten1', input='deconv_dropout0')
		model.add_output(name='deconv0_output', input='flatten1')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='valid'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Dropout(0.25), name='deconv_dropout1', input='deconv_relu1')

		if params is None:
			use_sat = True
		else:
			use_sat = params['use_saturation']
		if use_sat:
			model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')
			l = 'deconv_satlu'
		else:
			l = 'deconv_dropout1'
		#

		#model.add_node(Flatten(), name='flatten2', input='deconv_dropout1')
		model.add_node(Flatten(), name='flatten2', input=l)
		model.add_output(name='prediction_output', input='flatten2')

	elif model_name == 'mnist_prednet_discriminator0':

		model = Graph()

		model.add_input(name='proposed_frames', ndim=4)
		model.add_input(name='previous_frames', ndim=3)

		#model.add_node(CollapseTimesteps(ndim=2), name='collapse_time', input='previous_frames')
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='previous_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')

		# extract features
		model = add_mnist_cnn_feature_model0(model, 'proposed', 'proposed_frames', True)
		model = add_mnist_cnn_feature_model0(model, 'previous', 'reshape_frames', True)

		# format for LSTM
		model.add_node(Flatten(), name='flatten0', input='previous_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='expand_time', input='flatten0')
		if params['use_LSTM']:
			model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM', input='expand_time')
		else:
			model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM', input='expand_time')
		model.add_node(Dropout(0.5), name='LSTM_dropout', input='LSTM')

		model.add_node(Flatten(), name='flatten1', input='proposed_pool1')

		model.add_node(Dense(32*8*8+params['n_RNN_units'], 256), name='FC0', inputs=['LSTM_dropout', 'flatten1'], concat_axis=1)
		model.add_node(Dropout(0.5), name='FC0_dropout', input='FC0')
		model.add_node(Dense(256, 2), name='FC1', input='FC0_dropout')
		model.add_node(Activation('softmax'), name='softmax', input='FC1')
		model.add_output(name='output', input='softmax')


	elif model_name == 'mnist_prednet_G_to_D0':

		model = Graph()

		model.add_input(name='input_frames', ndim=3) #will have 2 frames per example
		model.add_input(name='random_input', ndim=2)

		#model.add_node(CollapseTimesteps(ndim=2), name='collapse_time', input='input_frames')
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='input_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')

		# get features from frames
		model = add_mnist_cnn_feature_model0(model, 'G', 'reshape_frames', True)

		# format for LSTM
		model.add_node(Flatten(), name='G_flatten', input='G_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='G_flatten')

		# LSTM
		if params['use_LSTM']:
			model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		else:
			model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		# Fully connected layer
		model.add_node(Dense(params['n_RNN_units']+128, 32*8*8), name='FC0', inputs=['LSTM_dropout0', 'random_input'], concat_axis=1)
		model.add_node(Activation('relu'), name='FC_relu0', input='FC0')
		model.add_node(Dropout(0.25), name='FC_dropout0', input='FC_relu0')
		model.add_node(Reshape(32, 8, 8), name='unflatten', input='FC_dropout0')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten')
		model.add_node(Convolution2D(32, 32, 4, 4, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='full'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')

		## Discriminator Network
		# get features from frames
		model = add_mnist_cnn_feature_model0(model, 'D', 'deconv_satlu', True)
		model.add_node(Flatten(), name='flatten1', input='D_pool1')

		if params['use_LSTM']:
			model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False, params_fixed=True), name='LSTM1', input='previous_features')
		else:
			model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False, params_fixed=True), name='LSTM1', input='previous_features')

		model.add_node(Dropout(0.5), name='LSTM1_dropout', input='LSTM1')
		model.add_node(Dense(32*8*8+params['n_RNN_units'], 256, params_fixed=True), name='FC1', inputs=['LSTM1_dropout', 'flatten1'], concat_axis=1)
		model.add_node(Dropout(0.5), name='FC1_dropout', input='FC1')
		model.add_node(Dense(256, 2, params_fixed=True), name='FC2', input='FC1_dropout')
		model.add_node(Activation('softmax'), name='softmax', input='FC2')
		model.add_output(name='output', input='softmax')

		if params['use_prediction_output']:
			model.add_output(name='prediction_output', input='deconv_satlu')


	elif model_name == 'mnist_prednet_G0':

		model = Graph()

		model.add_input(name='input_frames', ndim=3) #will have 2 frames per example
		model.add_input(name='random_input', ndim=2)

		#model.add_node(CollapseTimesteps(ndim=2), name='collapse_time', input='input_frames')
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='input_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')

		# get features from frames
		model = add_mnist_cnn_feature_model0(model, 'G', 'reshape_frames', True)

		# format for LSTM
		model.add_node(Flatten(), name='G_flatten', input='G_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='G_flatten')

		# LSTM
		model.add_node(LSTM(32*8*8, 1024, return_sequences=False), name='LSTM0', input='previous_features')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		# Fully connected layer
		model.add_node(Dense(1024+128, 32*8*8), name='FC0', inputs=['LSTM_dropout0', 'random_input'], concat_axis=1)
		model.add_node(Activation('relu'), name='FC_relu0', input='FC0')
		model.add_node(Dropout(0.25), name='FC_dropout0', input='FC_relu0')
		model.add_node(Reshape(32, 8, 8), name='unflatten', input='FC_dropout0')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten')
		model.add_node(Convolution2D(32, 32, 4, 4, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='full'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')

		model.add_output(name='output', input='deconv_satlu')


	elif model_name == 'mnist_prednet_G_to_D1':

		model = Graph()

		model.add_input(name='input_frames', ndim=3) #will have 2 frames per example
		#batch size must be 128
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='input_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')
		# encoder
		model = add_mnist_cnn_feature_model0(model, 'G', 'reshape_frames', True)
		# format for LSTM
		model.add_node(Flatten(), name='G_flatten', input='G_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='G_flatten')

		# LSTM
		if params['use_LSTM']:
			model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		else:
			model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		model.add_input(name='random_input', ndim=4)
		model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv0', input='random_input')
		model.add_node(Activation('relu'), name='randconv_relu0', input='randconv0')
		model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv1', input='randconv_relu0')
		model.add_node(Activation('relu'), name='randconv_relu1', input='randconv1')
		model.add_node(Flatten(), name='rand_flatten', input='randconv_relu1')

		# Fully connected layer and concat LSTM with random
		model.add_node(Dense(params['n_RNN_units']+(params['rand_nx']-4)**2, 1024), name='FC0', inputs=['LSTM_dropout0', 'rand_flatten'], concat_axis=1)
		model.add_node(Activation('relu'), name='FC_relu0', input='FC0')
		model.add_node(Dropout(0.25), name='FC_dropout0', input='FC_relu0')
		model.add_node(Dense(1024, 2048), name='FC1', input='FC_dropout0')
		model.add_node(Activation('relu'), name='FC_relu1', input='FC1')
		model.add_node(Dropout(0.25), name='FC_dropout1', input='FC_relu1')
		model.add_node(Reshape(32, 8, 8), name='unflatten', input='FC_dropout1')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten')
		model.add_node(Convolution2D(32, 32, 4, 4, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='full'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')

		if params['use_prediction_output']:
			model.add_output(name='prediction_output', input='deconv_satlu')

		## Discriminator Network
		# get features from generated frames
		model = add_mnist_cnn_feature_model0(model, 'D', 'deconv_satlu', True)
		model.add_node(Flatten(), name='flatten_proposed', input='D_pool1')

		model = add_mnist_prednet_D1(model, params, params_fixed=True)


	elif model_name == 'mnist_prednet_D1':

		model = Graph()

		model.add_input(name='proposed_frames', ndim=4)
		model = add_mnist_cnn_feature_model0(model, 'proposed', 'proposed_frames', True)
		model.add_node(Flatten(), name='flatten_proposed', input='proposed_pool1')

		# get encodings
		model.add_input(name='previous_frames', ndim=3)
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='previous_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')
		model = add_mnist_cnn_feature_model0(model, 'previous', 'reshape_frames', True)
		model.add_node(Flatten(), name='flatten_previous', input='previous_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='flatten_previous')

		model = add_mnist_prednet_D1(model, params, params_fixed=False)

		return model


	elif model_name == 'mnist_prednet_G1':

		model = Graph()

		model.add_input(name='input_frames', ndim=3) #will have 2 frames per example
		#batch size must be 128
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='input_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')
		# encoder
		model = add_mnist_cnn_feature_model0(model, 'G', 'reshape_frames', True)
		# format for LSTM
		model.add_node(Flatten(), name='G_flatten', input='G_pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='G_flatten')

		# LSTM
		if params['use_LSTM']:
			model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		else:
			model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False), name='LSTM0', input='previous_features')
		model.add_node(Dropout(0.25), name='LSTM_dropout0', input='LSTM0')

		model.add_input(name='random_input', ndim=4)
		model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv0', input='random_input')
		model.add_node(Activation('relu'), name='randconv_relu0', input='randconv0')
		model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv1', input='randconv_relu0')
		model.add_node(Activation('relu'), name='randconv_relu1', input='randconv1')
		model.add_node(Flatten(), name='rand_flatten', input='randconv_relu1')

		# Fully connected layer and concat LSTM with random
		model.add_node(Dense(params['n_RNN_units']+(params['rand_nx']-4)**2, 1024), name='FC0', inputs=['LSTM_dropout0', 'rand_flatten'], concat_axis=1)
		model.add_node(Activation('relu'), name='FC_relu0', input='FC0')
		model.add_node(Dropout(0.25), name='FC_dropout0', input='FC_relu0')
		model.add_node(Dense(1024, 2048), name='FC1', input='FC_dropout0')
		model.add_node(Activation('relu'), name='FC_relu1', input='FC1')
		model.add_node(Dropout(0.25), name='FC_dropout1', input='FC_relu1')
		model.add_node(Reshape(32, 8, 8), name='unflatten', input='FC_dropout1')

		# deconv net
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='unflatten')
		model.add_node(Convolution2D(32, 32, 4, 4, border_mode='valid'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv_relu0', input='deconv0')
		model.add_node(Dropout(0.25), name='deconv_dropout0', input='deconv_relu0')
		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv_dropout0')
		model.add_node(Convolution2D(1, 32, 3, 3, border_mode='full'), name='deconv1', input='unpool1')
		model.add_node(Activation('relu'), name='deconv_relu1', input='deconv1')
		model.add_node(Activation('satlu'), name='deconv_satlu', input='deconv_relu1')

		model.add_output(name='output', input='deconv_satlu')

	elif model_name == 'mnist_prednet_G_to_D2':

		# Generator
		model = mnist_prednet_G2(params['use_LSTM'], params['n_RNN_units'], params['rand_nx'], fix_encoder=True)
		#model.add_output(name='generated_output', input='G_proposed_features')
		#G_proposed_features will be (128, 32, 8, 8)

		if params['use_feature_output']:
			model.add_output(name='feature_output', input='G_proposed_features')

		# Discriminator
		model = add_mnist_prednet_D2(model, 'previous_features', 'G_proposed_features', params['use_LSTM'], params['n_RNN_units'], params_fixed=True, use_dropout=True)
		model.add_output(name='output', input='D_output')

		# deconv net
		model = add_mnist_cnn_deconv_model0(model, 'decoder', 'G_proposed_features', params_fixed=True, use_dropout=False)

		if params['use_prediction_output']:
			model.add_output(name='prediction_output', input='decoder_deconv_output')


	elif model_name == 'mnist_prednet_D2':

		model = Graph()

		# get encoding of proposed frames
		#model.add_input(name='proposed_frames', ndim=4)
		#model = add_mnist_cnn_feature_model0(model, 'proposed', 'proposed_frames', True)
		#model.add_node(Flatten(), name='proposed_features', input='proposed_pool1')

		# input will be (128, 32, 8, 8)
		model.add_input(name='proposed_features', ndim=4)

		# get encodings of previous frames
		model.add_input(name='previous_frames', ndim=3)
		model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='previous_frames')
		model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')
		model = add_mnist_cnn_feature_model0(model, 'previous', 'reshape_frames', True)
		model.add_node(Flatten(), name='flatten_previous', input='previous_feature_output')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='flatten_previous')

		# add Discriminator
		model = add_mnist_prednet_D2(model, 'previous_features', 'proposed_features', params['use_LSTM'], params['n_RNN_units'], params_fixed=False, use_dropout=True)

		model.add_output(name='output', input='D_output')

	elif model_name == 'mnist_deconv0':

		model = Graph()

		model.add_input(name='input', ndim=4)
		model = add_mnist_cnn_deconv_model0(model, '', 'input', params_fixed=False, use_dropout=False)

		model.add_output(name='output', input='_deconv_output')

	elif model_name == 'mnist_prednet_G2':

		# Generator
		model = mnist_prednet_G2(params['use_LSTM'], params['n_RNN_units'], params['rand_nx'], fix_encoder=True)

		model.add_output(name='output', input='G_proposed_features')

		model = add_mnist_cnn_deconv_model0(model, 'decoder', 'G_proposed_features', params_fixed=True, use_dropout=False)

		if params['use_prediction_output']:
			model.add_output(name='prediction_output', input='decoder_deconv_output')

	elif model_name == 'bouncing_ball_rnn':

		model = Graph()

		# input is (128, n_time_steps, frame_size**2)
		model.add_input(name='input_frames', ndim=3)

		model.add_node(FullReshape(params['batch_size']*params['n_timesteps'], params['frame_size']**2), name='collapse_time', input='input_frames')
		# (128*n_time_steps, frame_size**2)
		model.add_node(Reshape(1, params['frame_size'], params['frame_size']), name='frames', input='collapse_time')
		# (128*n_time_steps, 1, frame_size, frame_size)

		# encoder
		model.add_node(Convolution2D(8, 1, 3, 3, border_mode='full'), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
		# (128*n_time_steps, 8, 33, 33)

		model.add_node(Convolution2D(16, 8, 4, 4, border_mode='full'), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')
		# (128*n_time_steps, 16, 18, 18)

		model.add_node(Convolution2D(16, 16, 3, 3, border_mode='valid'), name='conv2', input='pool1')
		model.add_node(Activation('relu'), name='conv2_relu', input='conv2')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='features', input='conv2_relu')
		# (128*n_time_steps, 16, 8, 8)

		# format for LSTM
		model.add_node(Flatten(), name='flatten_features', input='features')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten_features')
		# (128, n_time_steps, 16*8*8)

		# LSTM
		model.add_node(LSTM(16*8*8, params['n_RNN_units'], return_sequences=False), name='RNN', input='previous_features')
		#model.add_node(Dropout(0.25), name='RNN_dropout', input='RNN')

		# FC
		model.add_node(Dense(params['n_RNN_units'], 16*8*8), name='FC', input='RNN')
		#model.add_node(Dropout(0.25), name='FC_dropout', input='FC')
		model.add_node(Reshape(16, 8, 8), name='FC_output', input='FC')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='FC_output')
		# (.., 16, 16, 16)
		model.add_node(Convolution2D(16, 16, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		# (.., 16, 14, 14)
		model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv0_relu')
		# (.., 16, 28, 28)
		model.add_node(Convolution2D(8, 16, 4, 4, border_mode='full'), name='deconv1', input='unpool1')
		# (.., 8, 31, 31)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool2', input='deconv1_relu')
		# (.., 8, 62, 62)
		model.add_node(Convolution2D(1, 8, 3, 3, border_mode='full'), name='deconv2', input='unpool2')
		# (.., 1, 64, 64)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')
		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu')

		model.add_output(name='output', input='predicted_frame')

	elif model_name=='bouncing_ball_autoencoder':

		model = Graph()

		# input is (batch_size, 1, 64, 64)
		model.add_input(name='input_frames', ndim=4)

		# encoder
		model.add_node(Convolution2D(8, 1, 3, 3, border_mode='full'), name='conv0', input='input_frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
		# (batch_size, 8, 33, 33)

		model.add_node(Convolution2D(16, 8, 4, 4, border_mode='full'), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')
		# (batch_size, 16, 18, 18)

		model.add_node(Convolution2D(16, 16, 3, 3, border_mode='valid'), name='conv2', input='pool1')
		model.add_node(Activation('relu'), name='conv2_relu', input='conv2')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='features', input='conv2_relu')


		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool0', input='features')
		# (.., 16, 16, 16)
		model.add_node(Convolution2D(16, 16, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		# (.., 16, 14, 14)
		model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool1', input='deconv0_relu')
		# (.., 16, 28, 28)
		model.add_node(Convolution2D(8, 16, 4, 4, border_mode='full'), name='deconv1', input='unpool1')
		# (.., 8, 31, 31)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UnPooling2D(unpoolsize=(2,2)), name='unpool2', input='deconv1_relu')
		# (.., 8, 62, 62)
		model.add_node(Convolution2D(1, 8, 3, 3, border_mode='full'), name='deconv2', input='unpool2')
		# (.., 1, 64, 64)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')
		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu')

		model.add_output(name='output', input='predicted_frame')

	elif model_name == 'bouncing_ball_rnn_simple':

		model = Graph()

		# input is (128, n_time_steps, frame_size**2)
		model.add_input(name='input_frames', ndim=3)

		# LSTM
		model.add_node(LSTM(4096, 4096, return_sequences=False), name='RNN', input='input_frames')

		model.add_node(Reshape(1, 64, 64), name='predicted_frame', input='RNN')

		model.add_output(name='output', input='predicted_frame')

	elif model_name == 'bouncing_ball_rnn_onelayer':

		model = Graph()

		# input is (128, n_time_steps, frame_size**2)
		model.add_input(name='input_frames', ndim=3)

		model.add_node(FullReshape(params['batch_size']*params['n_timesteps'], params['frame_size']**2), name='collapse_time', input='input_frames')
		# (128*n_time_steps, frame_size**2)
		model.add_node(Reshape(1, params['frame_size'], params['frame_size']), name='frames', input='collapse_time')
		# (128*n_time_steps, 1, frame_size, frame_size)

		# encoder
		model.add_node(Convolution2D(16, 1, 5, 5, border_mode='valid'), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(4,4)), name='features', input='conv0_relu')
		# (128*nt, 16, 15, 15)

		# format for LSTM
		model.add_node(Flatten(), name='flatten_features', input='features')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten_features')
		# (128, n_time_steps, 16*15*15)

		# LSTM
		model.add_node(LSTM(16*15*15, 16*15*15, return_sequences=False), name='RNN', input='previous_features')

		model.add_node(Reshape(16, 15, 15), name='pred_features', input='RNN')

		model.add_node(UpSample2D((4,4)), name='unpool0', input='pred_features')
		model.add_node(Convolution2D(1, 16, 5, 5, border_mode='full'), name='deconv0', input='unpool0')
		model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')
		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv0_relu')

		model.add_output(name='output', input='predicted_frame')

	elif model_name=='bouncing_ball_autoencoder_twolayer':

		model = Graph()

		# input is (batch_size, 1, 64, 64)
		model.add_input(name='input_frames', ndim=4)

		# encoder
		model.add_node(Convolution2D(32, 1, 5, 5, border_mode='full'), name='conv0', input='input_frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
		# (batch_size, 32, 34, 34)

		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')
		# (batch_size, 32, 16, 16)

		model.add_node(UpSample2D((2,2)), name='unpool0', input='pool1')
		# (.., 32, 32, 32)
		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
		# (.., 32, 30, 30)
		model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

		model.add_node(UpSample2D((2,2)), name='unpool1', input='deconv0_relu')
		# (.., 32, 60, 60)
		model.add_node(Convolution2D(1, 32, 5, 5, border_mode='full'), name='deconv1', input='unpool1')
		# (.., 1, 64, 64)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')
		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv1_relu')

		model.add_output(name='output', input='predicted_frame')

	elif model_name=='bouncing_ball_rnn_twolayer':

		model = Graph()

		# input is (batch_size, n_time_steps, frame_size**2)
		model.add_input(name='input_frames', ndim=3)

		#model.add_node(FullReshape(params['batch_size']*params['n_timesteps'], params['frame_size']**2), name='collapse_time', input='input_frames')
		model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames')
		# (128*n_time_steps, frame_size**2)
		model.add_node(Reshape(1, params['frame_size'], params['frame_size']), name='frames', input='collapse_time')
		# (128*n_time_steps, 1, frame_size, frame_size)


		if params['frame_size']==64:
			# encoder
			model.add_node(Convolution2D(32, 1, 5, 5, border_mode='full'), name='conv0', input='frames')
			model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
			# (batch_size, 32, 34, 34)
			model.add_node(Dropout(0.4), name='drop0', input='pool0')

			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='conv1', input='drop0')
			model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='poo11', input='conv1_relu')
			# (batch_size, 32, 16, 16)
			model.add_node(Dropout(0.4), name='drop1', input='')

			# format for LSTM
			model.add_node(Flatten(), name='flatten_features', input='features')
			model.add_node(ExpandTimesteps(ndim=3, batch_size=params['batch_size']), name='previous_features', input='flatten_features')
			# (128, n_time_steps, 32*16*16)

			# LSTM
			model.add_node(LSTM(32*16*16, 2048, return_sequences=False), name='RNN', input='previous_features')
			model.add_node(Reshape(32, 8, 8), name='pred_features', input='RNN')

			model.add_node(UpSample2D((4,4)), name='unpool0', input='pred_features')
			# (.., 32, 32, 32)
			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='deconv0', input='unpool0')
			# (.., 32, 30, 30)
			model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

			model.add_node(UpSample2D((2,2)), name='unpool1', input='deconv0_relu')
			# (.., 32, 60, 60)
			model.add_node(Convolution2D(1, 32, 5, 5, border_mode='full'), name='deconv1', input='unpool1')
			# (.., 1, 64, 64)
		else:
			# encoder
			model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0', input='frames')
			model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
			# (batch_size, 32, 17, 17)
			model.add_node(Dropout(0.4), name='drop0', input='pool0')

			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='conv1', input='drop0')
			model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')
			# (batch_size, 32, 7, 7)
			model.add_node(Dropout(0.4), name='drop1', input='pool1')

			# format for LSTM
			model.add_node(Flatten(), name='flatten_features', input='drop1')
			model.add_node(ExpandTimesteps(ndim=3, batch_size=params['batch_size']), name='previous_features', input='flatten_features')
			# (128, n_time_steps, 32*7*7)

			# LSTM
			model.add_node(LSTM(32*7*7, 1152, return_sequences=False), name='RNN', input='previous_features')
			model.add_node(Reshape(32, 6, 6), name='pred_features', input='RNN')

			model.add_node(UpSample2D((2,2)), name='unpool0', input='pred_features')
			# (.., 32, 12, 12)
			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full'), name='deconv0', input='unpool0')
			# (.., 32, 14, 14)
			model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

			model.add_node(UpSample2D((2,2)), name='unpool1', input='deconv0_relu')
			# (.., 32, 28, 28)
			model.add_node(Convolution2D(1, 32, 5, 5, border_mode='full'), name='deconv1', input='unpool1')
			# (.., 1, 32, 32)

		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')
		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv1_relu')

		model.add_output(name='output', input='predicted_frame')


	elif model_name=='bouncing_ball_rnn_twolayer_multsteps':

		model = Graph()


		model.add_input(name='input_frames', ndim=3)  # input is (batch_size, n_time_steps, frame_size**2)

		# PREPARE INPUT
		model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, frame_size**2)
		model.add_node(Reshape(1, params['frame_size'], params['frame_size']), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, frame_size, frame_size)


		# ENCODER
		model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0_0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu_0', input='conv0_0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0', input='conv0_relu_0')
		model.add_node(Dropout(0.4), name='drop0_0', input='pool0_0') # output: (batch_size, 32, 17, 17)

		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='conv1_0', input='drop0_0')
		model.add_node(Activation('relu'), name='conv1_relu_0', input='conv1_0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0', input='conv1_relu_0')
		model.add_node(Dropout(0.4), name='drop1_0', input='pool1_0') # output:  (batch_size, 32, 7, 7)


		# ENCODER FEATURES
		model.add_node(Flatten(), name='flatten_features_0', input='drop1_0')
		model.add_node(ExpandTimesteps(ndim=3, batch_size=params['batch_size']), name='previous_features', input='flatten_features_0')  # output:  (128, n_time_steps, 32*7*7)


		# LSTM
		model.add_node(LSTM(32*7*7, 1152, return_sequences=False), name='RNN_0', input='previous_features')
		model.add_node(Reshape(32, 6, 6), name='pred_features_0', input='RNN_0')


		# DECODER
		model.add_node(UpSample2D((2,2)), name='unpool0_0', input='pred_features_0')  # (.., 32, 12, 12)
		model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full'), name='deconv0_0', input='unpool0_0')  # (.., 32, 14, 14)
		model.add_node(Activation('relu'), name='deconv0_relu_0', input='deconv0_0')

		model.add_node(UpSample2D((2,2)), name='unpool1_0', input='deconv0_relu_0') # (.., 32, 28, 28)
		model.add_node(Convolution2D(1, 32, 5, 5, border_mode='full'), name='deconv1_0', input='unpool1_0')  # (.., 1, 32, 32)

		model.add_node(Activation('relu'), name='deconv1_relu_0', input='deconv1_0')
		model.add_node(Activation('satlu'), name='predicted_frame_0', input='deconv1_relu_0') # output:  (batch_size, 1, 32, 32)

		model.add_output(name='output_0', input='predicted_frame_0')

		for t in range(1, params['nt_predict']):

			model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', shared_weights_layer=model.nodes['conv0_0'], params_fixed=True), name='conv0_'+str(t), input='predicted_frame_'+str(t-1))
			model.add_node(Activation('relu'), name='conv0_relu_'+str(t), input='conv0_'+str(t))
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_'+str(t), input='conv0_relu_'+str(t))
			model.add_node(Dropout(0.4), name='drop0_'+str(t), input='pool0_'+str(t)) # output: (batch_size, 32, 17, 17)

			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', shared_weights_layer=model.nodes['conv1_0'], params_fixed=True), name='conv1_'+str(t), input='drop0_'+str(t))
			model.add_node(Activation('relu'), name='conv1_relu_'+str(t), input='conv1_'+str(t))
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_'+str(t), input='conv1_relu_'+str(t))
			model.add_node(Dropout(0.4), name='drop1_'+str(t), input='pool1_'+str(t)) # output:  (batch_size, 32, 7, 7)

			model.add_node(Flatten(), name='flatten_features_'+str(t), input='drop1_'+str(t)) # output:  (batch_size, 32*7*7)
			model.add_node(ExpandTimesteps(ndim=3, n_timesteps=1), name='features_'+str(t), input='flatten_features_'+str(t))  # output:  (128, n_time_steps, 32*7*7)

			model.add_node(LSTM(32*7*7, 1152, return_sequences=False, shared_weights_layer=model.nodes['RNN_0'], params_fixed=True), name='RNN_'+str(t), inputs=['previous_features', 'features_'+str(t)], merge_mode='concat', concat_axis=1)
			model.add_node(Reshape(32, 6, 6), name='pred_features_'+str(t), input='RNN_'+str(t))

			model.add_node(UpSample2D((2,2)), name='unpool0_'+str(t), input='pred_features_'+str(t))  # (.., 32, 12, 12)
			model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full', shared_weights_layer=model.nodes['deconv0_0'], params_fixed=True), name='deconv0_'+str(t), input='unpool0_'+str(t))  # (.., 32, 14, 14)
			model.add_node(Activation('relu'), name='deconv0_relu_'+str(t), input='deconv0_'+str(t))

			model.add_node(UpSample2D((2,2)), name='unpool1_'+str(t), input='deconv0_relu_'+str(t)) # (.., 32, 28, 28)
			model.add_node(Convolution2D(1, 32, 5, 5, border_mode='full', shared_weights_layer=model.nodes['deconv1_0'], params_fixed=True), name='deconv1_'+str(t), input='unpool1_'+str(t))  # (.., 1, 32, 32)

			model.add_node(Activation('relu'), name='deconv1_relu_'+str(t), input='deconv1_'+str(t))
			model.add_node(Activation('satlu'), name='predicted_frame_'+str(t), input='deconv1_relu_'+str(t)) # output:  (batch_size, 1, 32, 32)

			model.add_output(name='output_'+str(t), input='predicted_frame_'+str(t))

	elif model_name=='bouncing_ball_rnn_twolayer_multsteps_30x30':

		if 'nfilt' in params:
			nfilt = params['nfilt']
		else:
			nfilt = 32

		model = Graph()

		model.add_input(name='input_frames', ndim=3)  # input is (batch_size, n_time_steps, frame_size**2)

		# PREPARE INPUT
		model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, frame_size**2)
		model.add_node(Reshape(1, 30, 30), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, frame_size, frame_size)

		# ENCODER
		model.add_node(Convolution2D(nfilt, 1, 3, 3, border_mode='full'), name='conv0_0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu_0', input='conv0_0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0', input='conv0_relu_0')
		if params['encoder_dropout']:
			model.add_node(Dropout(0.4), name='drop0_0', input='pool0_0') # output: (batch_size, 32, 16, 16)
			prev = 'drop0_0'
		else:
			prev = 'pool0_0'

		model.add_node(Convolution2D(nfilt, nfilt, 3, 3, border_mode='valid'), name='conv1_0', input=prev)
		model.add_node(Activation('relu'), name='conv1_relu_0', input='conv1_0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0', input='conv1_relu_0')
		if params['encoder_dropout']:
			model.add_node(Dropout(0.4), name='drop1_0', input='pool1_0') # output:  (batch_size, 32, 7, 7)
			prev = 'drop1_0'
		else:
			prev = 'pool1_0'


		# ENCODER FEATURES
		model.add_node(Flatten(), name='flatten_features_0', input=prev)
		model.add_node(ExpandTimesteps(ndim=3, batch_size=params['batch_size']), name='previous_features', input='flatten_features_0')  # output:  (128, n_time_steps, 32*7*7)


		# LSTM
		model.add_node(LSTM(nfilt*7*7, nfilt*7*7, return_sequences=False), name='RNN_0', input='previous_features')
		if params['LSTM_dropout']:
			model.add_node(Dropout(0.4), name='rnndrop_0', input='RNN_0')
			prev = 'rnndrop_0'
		else:
			prev = 'RNN_0'
		model.add_node(Reshape(nfilt, 7, 7), name='pred_features_0', input=prev)


		# DECODER
		model.add_node(UpSample2D((2,2)), name='unpool0_0', input='pred_features_0')  # (.., 32, 14, 14)
		model.add_node(Convolution2D(nfilt, nfilt, 3, 3, border_mode='full'), name='deconv0_0', input='unpool0_0')  # (.., 32, 16, 16)
		model.add_node(Activation('relu'), name='deconv0_relu_0', input='deconv0_0')

		model.add_node(UpSample2D((2,2)), name='unpool1_0', input='deconv0_relu_0') # (.., 32, 32, 32)
		model.add_node(Convolution2D(1, nfilt, 3, 3, border_mode='valid'), name='deconv1_0', input='unpool1_0')  # (.., 1, 30, 30)

		model.add_node(Activation('relu'), name='deconv1_relu_0', input='deconv1_0')
		model.add_node(Activation('satlu'), name='predicted_frame_0', input='deconv1_relu_0') # output:  (batch_size, 1, 30, 30)

		model.add_output(name='output_0', input='predicted_frame_0')

		for t in range(1, params['nt_predict']):

			model.add_node(Convolution2D(nfilt, 1, 3, 3, border_mode='full', shared_weights_layer=model.nodes['conv0_0'], params_fixed=True), name='conv0_'+str(t), input='predicted_frame_'+str(t-1))
			model.add_node(Activation('relu'), name='conv0_relu_'+str(t), input='conv0_'+str(t))
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_'+str(t), input='conv0_relu_'+str(t))
			if params['encoder_dropout']:
				model.add_node(Dropout(0.4), name='drop0_'+str(t), input='pool0_'+str(t)) # output: (batch_size, 32, 17, 17)
				prev = 'drop0_'+str(t)
			else:
				prev = 'pool0_'+str(t)

			model.add_node(Convolution2D(nfilt, nfilt, 3, 3, border_mode='valid', shared_weights_layer=model.nodes['conv1_0'], params_fixed=True), name='conv1_'+str(t), input=prev)
			model.add_node(Activation('relu'), name='conv1_relu_'+str(t), input='conv1_'+str(t))
			model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_'+str(t), input='conv1_relu_'+str(t))
			if params['encoder_dropout']:
				model.add_node(Dropout(0.4), name='drop1_'+str(t), input='pool1_'+str(t)) # output:  (batch_size, 32, 7, 7)
				prev = 'drop1_'+str(t)
			else:
				prev = 'pool1_'+str(t)

			model.add_node(Flatten(), name='flatten_features_'+str(t), input=prev) # output:  (batch_size, 32*7*7)
			model.add_node(ExpandTimesteps(ndim=3, n_timesteps=1), name='features_'+str(t), input='flatten_features_'+str(t))  # output:  (128, n_time_steps, 32*7*7)

			model.add_node(LSTM(nfilt*7*7, nfilt*7*7, return_sequences=False, shared_weights_layer=model.nodes['RNN_0'], params_fixed=True), name='RNN_'+str(t), inputs=['previous_features', 'features_'+str(t)], merge_mode='concat', concat_axis=1)
			if params['LSTM_dropout']:
				model.add_node(Dropout(0.4), name='rnndrop_'+str(t), input='RNN_'+str(t))
				prev = 'rnndrop_'+str(t)
			else:
				prev = 'RNN_'+str(t)
			model.add_node(Reshape(nfilt, 7, 7), name='pred_features_'+str(t), input=prev)

			model.add_node(UpSample2D((2,2)), name='unpool0_'+str(t), input='pred_features_'+str(t))  # (.., 32, 12, 12)
			model.add_node(Convolution2D(nfilt, nfilt, 3, 3, border_mode='full', shared_weights_layer=model.nodes['deconv0_0'], params_fixed=True), name='deconv0_'+str(t), input='unpool0_'+str(t))  # (.., 32, 14, 14)
			model.add_node(Activation('relu'), name='deconv0_relu_'+str(t), input='deconv0_'+str(t))

			model.add_node(UpSample2D((2,2)), name='unpool1_'+str(t), input='deconv0_relu_'+str(t)) # (.., 32, 28, 28)
			model.add_node(Convolution2D(1, nfilt, 3, 3, border_mode='valid', shared_weights_layer=model.nodes['deconv1_0'], params_fixed=True), name='deconv1_'+str(t), input='unpool1_'+str(t))  # (.., 1, 32, 32)

			model.add_node(Activation('relu'), name='deconv1_relu_'+str(t), input='deconv1_'+str(t))
			model.add_node(Activation('satlu'), name='predicted_frame_'+str(t), input='deconv1_relu_'+str(t)) # output:  (batch_size, 1, 32, 32)

			model.add_output(name='output_'+str(t), input='predicted_frame_'+str(t))


	elif model_name == 'vgg_16':

		model = Graph()

		model.add_input(name='input', ndim=4)

		model.add_node(Convolution2D(64, 3, 3, 3, border_mode='same'), name='conv1_1', input='input')
		model.add_node(Activation('relu'), name='relu1_1', input='conv1_1')
		model.add_node(Convolution2D(64, 64, 3, 3, border_mode='same'), name='conv1_2', input='relu1_1')
		model.add_node(Activation('relu'), name='relu1_2', input='conv1_2')
		model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool1', input='relu1_2')

		model.add_node(Convolution2D(128, 64, 3, 3, border_mode='same'), name='conv2_1', input='pool1')
		model.add_node(Activation('relu'), name='relu2_1', input='conv2_1')
		model.add_node(Convolution2D(128, 128, 3, 3, border_mode='same'), name='conv2_2', input='relu2_1')
		model.add_node(Activation('relu'), name='relu2_2', input='conv2_2')
		model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool2', input='relu2_2')

		model.add_node(Convolution2D(256, 128, 3, 3, border_mode='same'), name='conv3_1', input='pool2')
		model.add_node(Activation('relu'), name='relu3_1', input='conv3_1')
		model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='conv3_2', input='relu3_1')
		model.add_node(Activation('relu'), name='relu3_2', input='conv3_2')
		model.add_node(Convolution2D(256, 256, 3, 3, border_mode='same'), name='conv3_3', input='relu3_2')
		model.add_node(Activation('relu'), name='relu3_3', input='conv3_3')
		model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool3', input='relu3_3')

		model.add_node(Convolution2D(512, 256, 3, 3, border_mode='same'), name='conv4_1', input='pool3')
		model.add_node(Activation('relu'), name='relu4_1', input='conv4_1')
		model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv4_2', input='relu4_1')
		model.add_node(Activation('relu'), name='relu4_2', input='conv4_2')
		model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv4_3', input='relu4_2')
		model.add_node(Activation('relu'), name='relu4_3', input='conv4_3')
		model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool4', input='relu4_3')

		model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_1', input='pool4')
		model.add_node(Activation('relu'), name='relu5_1', input='conv5_1')
		model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_2', input='relu5_1')
		model.add_node(Activation('relu'), name='relu5_2', input='conv5_2')
		model.add_node(Convolution2D(512, 512, 3, 3, border_mode='same'), name='conv5_3', input='relu5_2')
		model.add_node(Activation('relu'), name='relu5_3', input='conv5_3')
		model.add_node(MaxPooling2D(poolsize=(2, 2)), name='pool5', input='relu5_3')

		model.add_node(Flatten(), name='flatten', input='pool5')
		model.add_node(Dense(512*7*7, 4096), name='fc6', input='flatten')
		model.add_node(Activation('relu'), name='relu6', input='fc6')
		model.add_node(Dense(4096, 4096), name='fc7', input='relu6')
		model.add_node(Activation('relu'), name='relu7', input='fc7')

		model.add_node(Dense(4096, 1000), name='fc8', input='relu7')
		model.add_node(Activation('softmax'), name='prob', input='fc8')

		model.add_output(name='output', input='prob')

	elif model_name == 'facegen_rotation_prednet' or model_name == 'facegen_rotation_prednet_flattened':

		model = Graph()

		model.add_input(name='input_frames', ndim=3)  # input is (batch_size, n_time_steps, 150**2)

		model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, 150**2)
		model.add_node(Reshape(1, 150, 150), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, 150, 150)

		model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid'), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
		#model.add_node(Dropout(0.4), name='drop0', input='pool0') # output: (batch_size*n_time_steps, 64, 47, 47)

		model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid'), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')
		#model.add_node(Dropout(0.4), name='drop1', input='pool1') # output:  (batch_size*n_time_steps, 64, 21, 21)

		model.add_node(Convolution2D(32, 64, 5, 5, border_mode='valid'), name='conv2', input='pool1')
		model.add_node(Activation('relu'), name='conv2_relu', input='conv2')
		model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool2', input='conv2_relu')
		#model.add_node(Dropout(0.4), name='drop2', input='pool2') # output:  (batch_size*n_time_steps, 32, 8, 8)

		#model.add_node(Flatten(), name='flatten', input='drop2')
		#model.add_node(Dense(32*8*8, 1024), name='fc0', input='flatten') # output:  (batch_size*n_time_steps, 1024)

		model.add_node(Flatten(), name='flatten', input='pool2')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten')  # output:  (128, n_time_steps, 32*7*7)
		model.add_node(LSTM(32*15*15, 1024, return_sequences=False), name='RNN', input='previous_features')

		model.add_node(Dense(1024, 32*14*14), name='fc_decoder', input='RNN')
		model.add_node(Activation('relu'), name='fc_decoder_relu', input='fc_decoder')
		model.add_node(Reshape(32, 14, 14), name='decoder_reshape', input='fc_decoder_relu')

		model.add_node(UpSample2D((2,2)), name='unpool0', input='decoder_reshape')  # (.., 32, 28, 28)
		model.add_node(Convolution2D(64, 32, 7, 7, border_mode='full'), name='deconv0', input='unpool0')  # (.., 32, 34, 34)
		model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

		model.add_node(UpSample2D((2,2)), name='unpool1', input='deconv0_relu')  # (.., 32, 68, 68)
		model.add_node(Convolution2D(64, 64, 5, 5, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UpSample2D((2,2)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
		model.add_node(Convolution2D(1, 32, 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')

		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

		if model_name=='facegen_rotation_prednet_flattened':
			model.add_node(Flatten(), name='predicted_frame_flattened', input='predicted_frame')
			model.add_output(name='output', input='predicted_frame_flattened')
		else:
			model.add_output(name='output', input='predicted_frame')

	elif model_name == 'facegen_rotation_prednet_twolayer' or model_name == 'facegen_rotation_prednet_twolayer_flattened':

		model = Graph()

		model.add_input(name='input_frames', ndim=3)  # input is (batch_size, n_time_steps, 150**2)

		model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, 150**2)
		model.add_node(Reshape(1, 150, 150), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, 150, 150)

		model.add_node(Convolution2D(params['num_filt'], 1, 5, 5, border_mode='valid'), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='conv0_relu')
		if params['use_encoder_drop0']:
			model.add_node(Dropout(0.4), name='drop0', input='pool0') # output: (batch_size*n_time_steps, 64, 47, 47)
			prev_node = 'drop0'
		else:
			prev_node = 'pool0'

		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 5, 5, border_mode='valid'), name='conv1', input=prev_node)
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')
		if params['use_encoder_drop1']:
			model.add_node(Dropout(0.4), name='drop1', input='pool1') # output:  (batch_size*n_time_steps, 64, 21, 21)
			prev_node = 'drop1'
		else:
			prev_node = 'pool1'

		#model.add_node(Flatten(), name='flatten', input='drop2')
		#model.add_node(Dense(32*8*8, 1024), name='fc0', input='flatten') # output:  (batch_size*n_time_steps, 1024)

		model.add_node(Flatten(), name='flatten', input=prev_node)
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten')  # output:  (128, n_time_steps, 32*7*7)
		model.add_node(LSTM(params['num_filt']*14*14, 1024, return_sequences=False), name='RNN', input='previous_features')

		model.add_node(Dense(1024, params['num_filt']*14*14), name='fc_decoder', input='RNN')
		model.add_node(Activation('relu'), name='fc_decoder_relu', input='fc_decoder')
		if params['use_dense_drop']:
			model.add_node(Dropout(0.4), name='dense_drop', input='fc_decoder_relu')
			prev_node = 'dense_drop'
		else:
			prev_node = 'fc_decoder_relu'
		model.add_node(Reshape(params['num_filt'], 14, 14), name='decoder_reshape', input=prev_node)

		model.add_node(UpSample2D((3,3)), name='unpool1', input='decoder_reshape')  # (.., 32, 68, 68)
		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 7, 7, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UpSample2D((3,3)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
		model.add_node(Convolution2D(1, params['num_filt'], 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')

		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

		if model_name=='facegen_rotation_prednet_twolayer_flattened':
			model.add_node(Flatten(), name='predicted_frame_flattened', input='predicted_frame')
			model.add_output(name='output', input='predicted_frame_flattened')
		else:
			model.add_output(name='output', input='predicted_frame')


	elif model_name == 'facegen_rotation_autoencoder':

		model = Graph()

		model.add_input(name='input_frames', ndim=4)


		model.add_node(Convolution2D(params['num_filt'], 1, 5, 5, border_mode='valid'), name='conv0', input='input_frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='conv0_relu')
		if params['use_encoder_drop0']:
			model.add_node(Dropout(0.4), name='drop0', input='pool0') # output: (batch_size*n_time_steps, 64, 47, 47)
			prev_node = 'drop0'
		else:
			prev_node = 'pool0'

		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 5, 5, border_mode='valid'), name='conv1', input=prev_node)
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')
		if params['use_encoder_drop1']:
			model.add_node(Dropout(0.4), name='drop1', input='pool1') # output:  (batch_size*n_time_steps, 64, 21, 21)
			prev_node = 'drop1'
		else:
			prev_node = 'pool1'

		# model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 5, 5, border_mode='valid'), name='conv2', input='pool1')
		# model.add_node(Activation('relu'), name='conv2_relu', input='conv2')
		# model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool2', input='conv2_relu')
		#model.add_node(Dropout(0.4), name='drop2', input='pool2') # output:  (batch_size*n_time_steps, 32, 15, 15)

		model.add_node(Flatten(), name='flatten', input=prev_node)

		model.add_node(Dense(params['num_filt']*14*14, params['n_FC']), name='fc_encoder', input='flatten')
		model.add_node(Activation('tanh'), name='fc_encoder_relu', input='fc_encoder')

		model.add_node(Dense(params['n_FC'], params['num_filt']*14*14), name='fc_decoder', input='fc_encoder_relu')
		model.add_node(Activation('relu'), name='fc_decoder_relu', input='fc_decoder')
		if params['use_dense_drop']:
			model.add_node(Dropout(0.4), name='dense_drop', input='fc_decoder_relu')
			prev_node = 'dense_drop'
		else:
			prev_node = 'fc_decoder_relu'
		model.add_node(Reshape(params['num_filt'], 14, 14), name='decoder_reshape', input=prev_node)

		# model.add_node(UpSample2D((2,2)), name='unpool0', input='decoder_reshape')  # (.., 32, 30, 30)
		# model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 5, 5, border_mode='full'), name='deconv0', input='unpool0')  # (.., 32, 34, 34)
		# model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')

		#model.add_node(UpSample2D((5,5)), name='unpool1', input='pool2')  # (.., 32, 75, 75)
		#model.add_node(Convolution2D(64, 64, 4, 4, border_mode='valid'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		#model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UpSample2D((3,3)), name='unpool1', input='decoder_reshape')  # (.., params['num_filt'], 68, 68)
		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 7, 7, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UpSample2D((3,3)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
		model.add_node(Convolution2D(1, params['num_filt'], 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')

		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

		model.add_output(name='output', input='predicted_frame')

	elif model_name == 'facegen_autoencoder':

		model = Graph()

		model.add_input(name='input_frames', ndim=4)  # input is (batch_size, 1, 150, 150)

		model.add_node(Convolution2D(params['num_filt'], 1, 5, 5, border_mode='valid'), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(4,4)), name='pool0', input='conv0_relu')
		#model.add_node(Dropout(0.4), name='drop0', input='pool0') # output: (batch_size*n_time_steps, 64, 47, 47)

		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 5, 5, border_mode='valid'), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')
		#model.add_node(Dropout(0.4), name='drop1', input='pool1') # output:  (batch_size*n_time_steps, 64, 21, 21)

		#model.add_node(Flatten(), name='flatten', input='drop2')
		#model.add_node(Dense(32*8*8, 1024), name='fc0', input='flatten') # output:  (batch_size*n_time_steps, 1024)

		model.add_node(Flatten(), name='flatten', input='pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten')  # output:  (128, n_time_steps, 32*7*7)
		model.add_node(LSTM(params['num_filt']*14*14, 1024, return_sequences=False), name='RNN', input='previous_features')

		model.add_node(Dense(1024, params['num_filt']*14*14), name='fc_decoder', input='RNN')
		model.add_node(Reshape(params['num_filt'], 14, 14), name='decoder_reshape', input='fc_decoder')

		model.add_node(UpSample2D((3,3)), name='unpool1', input='decoder_reshape')  # (.., 32, 68, 68)
		model.add_node(Convolution2D(params['num_filt'], params['num_filt'], 7, 7, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		model.add_node(UpSample2D((3,3)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
		model.add_node(Convolution2D(1, params['num_filt'], 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
		model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')

		model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

		model.add_output(name='output', input='predicted_frame')


	elif model_name == 'vgg_16_test':

		model = Graph()

		model.add_input(name='input', ndim=4)

		model.add_node(Convolution2D(64, 3, 3, 3, border_mode='valid'), name='conv1_1', input='input')

		model.add_output(name='output', input='conv1_1')

	elif model_name == 'conv_test':

		model = Graph()

		model.add_input(name='input', ndim=4)

		model.add_node(Convolution2D(1, 3, 3, 3, border_mode='valid'), name='conv', input='input')

		model.add_output(name='output', input='conv')


	elif model_name == 'facegen_rotation_prednet_twolayer_D':

		model = Graph()

		model.add_input(name='proposed_input', ndim=4)

		model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=params['c0_shared_layer_D'], params_fixed=params['c0_params_fixed_D']), name='conv0_D', input='predicted_frame')
		model.add_node(Activation('relu'), name='conv0_relu_D', input='conv0_D')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0_D', input='conv0_relu_D')

		model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=params['c1_shared_layer_D'], params_fixed=params['c1_params_fixed_D']), name='conv1_D', input='pool0_D')
		model.add_node(Activation('relu'), name='conv1_relu_D', input='conv1_D')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1_D', input='conv1_relu_D')


		model.add_input(name='previous_frames', ndim=3)
		model.add_node(CollapseTimesteps(2), name='collapse_time', input='previous_frames') # output: (batch_size*n_time_steps, 150**2)
		model.add_node(Reshape(1, 150, 150), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, 150, 150)

		model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=params['c0_shared_layer'], params_fixed=params['c0_params_fixed']), name='conv0', input='frames')
		model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='conv0_relu')

		model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=params['c1_shared_layer'], params_fixed=params['c1_params_fixed']), name='conv1', input='pool0')
		model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')

		model.add_node(Flatten(), name='flatten', input='pool1')
		model.add_node(ExpandTimesteps(ndim=3, n_timesteps=params['n_timesteps']), name='previous_features', input='flatten')  # output:  (128, n_time_steps, 32*7*7)
		model.add_node(LSTM(64*14*14, 1024, return_sequences=False, shared_weights_layer=params['RNN_shared_layer'], params_fixed=params['RNN_params_fixed']), name='RNN', input='previous_features')


		model.add_node(Dense(64*14*14+1024, 2048), name='fc0_D', inputs=['conv1_relu_D', 'RNN'], merge_mode='concat')
		model.add_node(Activation('relu'), name='fc0_relu_D', input='fc0_D')
		model.add_node(Dense(2048, 512), name='fc1_D', input='fc0_relu_D')
		model.add_node(Activation('relu'), name='fc1_relu_D', input='fc1_D')
		model.add_node(Dense(512, 2), name='fc2_D', input='fc1_relu_D')
		model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
		model.add_output(name='output', input='D_softmax')




	return model



def initialize_GAN_models(G_model, D_model, G_params, D_params):

	if G_model=='facegen_rotation_prednet_twolayer_G_to_D' and D_model=='facegen_rotation_prednet_twolayer_D':

		G_model = Graph()

		G_model.add_input(name='previous_frames', ndim=3)  # input is (batch_size, n_time_steps, 150**2)
		G_model.add_node(CollapseTimesteps(2), name='collapse_time', input='previous_frames') # output: (batch_size*n_time_steps, 150**2)
		G_model.add_node(Reshape(1, 150, 150), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, 150, 150)

		# ENCODER for previous frames
		G_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid'), name='conv0', input='frames')
		G_model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='conv0_relu')
		G_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid'), name='conv1', input='pool0')
		G_model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')

		# RNN for generator
		G_model.add_node(Flatten(), name='flatten', input='pool1')
		G_model.add_node(ExpandTimesteps(ndim=3, n_timesteps=G_params['n_timesteps']), name='previous_features', input='flatten')  # output:  (128, n_time_steps, 32*7*7)
		G_model.add_node(LSTM(64*14*14, 1024, return_sequences=False), name='RNN', input='previous_features')

		if G_params['use_rand_input']:
			# add in noise
			G_model.add_input(name='random_input', ndim=2)
			# DECODER
			G_model.add_node(Dense(1024+G_params['rand_size'], 64*14*14), name='fc_decoder', inputs=['RNN', 'random_input'], merge_mode='concat', concat_axis=1)
		else:
			G_model.add_node(Dense(1024, 64*14*14), name='fc_decoder', input='RNN')
		G_model.add_node(Activation('relu'), name='fc_decoder_relu', input='fc_decoder')
		G_model.add_node(Reshape(64, 14, 14), name='decoder_reshape', input='fc_decoder_relu')

		#if G_params['use_feature_output']:
		#	G_model.add_output(name='feature_output', input='decoder_reshape')

		G_model.add_node(UpSample2D((3,3)), name='unpool1', input='decoder_reshape')  # (.., 32, 68, 68)
		G_model.add_node(Convolution2D(64, 64, 7, 7, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
		G_model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

		G_model.add_node(UpSample2D((3,3)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
		G_model.add_node(Convolution2D(1, 64, 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
		G_model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')
		G_model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

		if G_params['use_pixel_output']:
			if G_params['pixel_output_flattened']:
				G_model.add_node(Flatten(), name='pixel_output_flattened', input='predicted_frame')
				G_model.add_output(name='pixel_output', input='pixel_output_flattened')
			else:
				G_model.add_output(name='pixel_output', input='predicted_frame')



		#####
		# Define Discriminator
		#####

		D_model = Graph()

		D_model.add_input(name='proposed_frames', ndim=4)

		# ENCODER for Discriminator
		if D_params['share_encoder']:
			D_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv0'], params_fixed=D_params['encoder_params_fixed']), name='conv0_D', input='proposed_frames')
		else:
			D_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid'), name='conv0_D', input='proposed_frames')
		D_model.add_node(Activation('relu'), name='conv0_relu_D', input='conv0_D')
		D_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0_D', input='conv0_relu_D')

		if D_params['share_encoder']:
			D_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv1'], params_fixed=D_params['encoder_params_fixed']), name='conv1_D', input='pool0_D')
		else:
			D_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid'), name='conv1_D', input='pool0_D')
		D_model.add_node(Activation('relu'), name='conv1_relu_D', input='conv1_D')
		D_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1_D', input='conv1_relu_D')
		D_model.add_node(Flatten(), name='proposed_flattened', input='pool1_D')

		# PREVIOUS FRAMES for Discriminator
		D_model.add_input(name='previous_frames', ndim=3)
		D_model.add_node(CollapseTimesteps(2), name='collapse_time', input='previous_frames') # output: (batch_size*n_time_steps, 150**2)
		D_model.add_node(Reshape(1, 150, 150), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, 150, 150)

		if D_params['share_encoder']:
			D_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv0'], params_fixed=True), name='conv0', input='frames')
		else:
			D_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_D'], params_fixed=True), name='conv0', input='frames')
		D_model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
		D_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='conv0_relu')

		if D_params['share_encoder']:
			D_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv1'], params_fixed=True), name='conv1', input='pool0')
		else:
			D_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv1_D'], params_fixed=True), name='conv1', input='pool0')
		D_model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
		D_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='conv1_relu')

		# RNN for Discriminator
		D_model.add_node(Flatten(), name='flatten', input='pool1')
		D_model.add_node(ExpandTimesteps(ndim=3, n_timesteps=D_params['n_timesteps']), name='previous_features', input='flatten')
		if D_params['share_RNN']:
			RNN_shared_layer = G_model.nodes['RNN']
			RNN_params_fixed = D_params['RNN_params_fixed']
			n_LSTM = 1024
		else:
			RNN_shared_layer = None
			RNN_params_fixed = False
			n_LSTM = 1024
		if 'n_LSTM' in D_params:
			n_LSTM = D_params['n_LSTM']
		D_model.add_node(LSTM(64*14*14, n_LSTM, return_sequences=False, shared_weights_layer=RNN_shared_layer, params_fixed=RNN_params_fixed), name='RNN', input='previous_features')

		if 'RNN_mult' in D_params:
			k = D_params['RNN_mult']
		else:
			k = 1.0

		D_model.add_node(Scalar_Multiply(k), name='RNN_scaled', input='RNN')

		if 'fusion_type' in D_params:
			fusion_type = D_params['fusion_type']
		else:
			fusion_type = 'early'

		if 'use_fc_precat' in D_params:
			use_fc_precat = D_params['use_fc_precat']
		else:
			use_fc_precat = False
		if fusion_type=='early':

			if use_fc_precat:
				D_model.add_node(Dense(64*14*14, D_params['fc_precat_size']), name='fc_precat', input='proposed_flattened')
				D_model.add_node(Activation('relu'), name='fc_precat_relu', input='fc_precat')
				D_model.add_node(Dense(D_params['fc_precat_size']+n_LSTM, 1024), name='fc0_D', inputs=['fc_precat_relu', 'RNN_scaled'], merge_mode='concat', concat_axis=1)
			else:
				D_model.add_node(Dense(64*14*14+n_LSTM, 1024), name='fc0_D', inputs=['proposed_flattened', 'RNN_scaled'], merge_mode='concat', concat_axis=1)
			D_model.add_node(Activation('relu'), name='fc0_relu_D', input='fc0_D')
			D_model.add_node(Dense(1024, 256), name='fc1_D', input='fc0_relu_D')
			D_model.add_node(Activation('relu'), name='fc1_relu_D', input='fc1_D')
			D_model.add_node(Dense(256, 2), name='fc2_D', input='fc1_relu_D')
			D_model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
			D_model.add_output(name='output', input='D_softmax')

		elif fusion_type=='late':
			D_model.add_node(Dense(n_LSTM, 512), name='fc0_RNN' , input='RNN_scaled')
			D_model.add_node(Activation('relu'), name='fc0_RNN_relu', input='fc0_RNN')
			D_model.add_node(Dense(512, 256), name='fc1_RNN' , input='fc0_RNN_relu')
			D_model.add_node(Activation('relu'), name='fc1_RNN_relu', input='fc1_RNN')

			D_model.add_node(Dense(64*14*14, 1024), name='fc0_prop' , input='proposed_flattened')
			D_model.add_node(Activation('relu'), name='fc0_prop_relu', input='fc0_prop')
			D_model.add_node(Dense(1024, 256), name='fc1_prop' , input='fc0_prop_relu')
			D_model.add_node(Activation('relu'), name='fc1_prop_relu', input='fc1_prop')

			D_model.add_node(Dense(512, 128), name='fc0_fus' , inputs=['fc1_prop_relu', 'fc1_RNN_relu'], merge_mode='concat', concat_axis=1)
			D_model.add_node(Activation('relu'), name='fc0_fus_relu', input='fc0_fus')
			D_model.add_node(Dense(128, 2), name='fc1_fus' , input='fc0_fus_relu')
			D_model.add_node(Activation('softmax'), name='D_softmax', input='fc1_fus')
			D_model.add_output(name='output', input='D_softmax')


		######
		# Finish Defining Generator
		#####

		# ENCODER for Discriminator
		if D_params['share_encoder']:
			G_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv0'], params_fixed=True), name='conv0_D', input='predicted_frame')
		else:
			G_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_D'], params_fixed=True), name='conv0_D', input='predicted_frame')
		G_model.add_node(Activation('relu'), name='conv0_relu_D', input='conv0_D')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0_D', input='conv0_relu_D')

		if D_params['share_encoder']:
			G_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=G_model.nodes['conv1'], params_fixed=True), name='conv1_D', input='pool0_D')
		else:
			G_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv1_D'], params_fixed=True), name='conv1_D', input='pool0_D')
		G_model.add_node(Activation('relu'), name='conv1_relu_D', input='conv1_D')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1_D', input='conv1_relu_D')
		G_model.add_node(Flatten(), name='proposed_flattened', input='pool1_D')

		if D_params['share_RNN']:
			fc0_D_RNN_input = 'RNN'
		else:
			G_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_D'], params_fixed=True), name='conv0_D_RNN', input='frames')
			G_model.add_node(Activation('relu'), name='conv0_relu_D_RNN', input='conv0_D_RNN')
			G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0_D_RNN', input='conv0_relu_D_RNN')
			G_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv1_D'], params_fixed=True), name='conv1_D_RNN', input='pool0_D_RNN')
			G_model.add_node(Activation('relu'), name='conv1_relu_D_RNN', input='conv1_D_RNN')
			G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1_D_RNN', input='conv1_relu_D_RNN')
			G_model.add_node(Flatten(), name='flatten_D_RNN', input='pool1_D_RNN')
			G_model.add_node(ExpandTimesteps(ndim=3, n_timesteps=D_params['n_timesteps']), name='previous_features_D_RNN', input='flatten_D_RNN')

			G_model.add_node(LSTM(64*14*14, n_LSTM, return_sequences=False, shared_weights_layer=D_model.nodes['RNN'], params_fixed=True), name='RNN_D', input='previous_features_D_RNN')
			fc0_D_RNN_input = 'RNN_D'

		G_model.add_node(Scalar_Multiply(k), name='RNN_scaled', input=fc0_D_RNN_input)

		if fusion_type=='early':

			if use_fc_precat:
				G_model.add_node(Dense(64*14*14, D_params['fc_precat_size'], shared_weights_layer=D_model.nodes['fc_precat'], params_fixed=True), name='fc_precat', input='proposed_flattened')
				G_model.add_node(Activation('relu'), name='fc_precat_relu', input='fc_precat')
				G_model.add_node(Dense(D_params['fc_precat_size']+n_LSTM, 1024, shared_weights_layer=D_model.nodes['fc0_D'], params_fixed=True), name='fc0_D', inputs=['fc_precat_relu', 'RNN_scaled'], merge_mode='concat', concat_axis=1)
			else:
				G_model.add_node(Dense(64*14*14+n_LSTM, 1024, shared_weights_layer=D_model.nodes['fc0_D'], params_fixed=True), name='fc0_D', inputs=['proposed_flattened', 'RNN_scaled'], merge_mode='concat', concat_axis=1)

			G_model.add_node(Activation('relu'), name='fc0_relu_D', input='fc0_D')
			G_model.add_node(Dense(1024, 256, shared_weights_layer=D_model.nodes['fc1_D'], params_fixed=True), name='fc1_D', input='fc0_relu_D')
			G_model.add_node(Activation('relu'), name='fc1_relu_D', input='fc1_D')
			G_model.add_node(Dense(256, 2, shared_weights_layer=D_model.nodes['fc2_D'], params_fixed=True), name='fc2_D', input='fc1_relu_D')
			G_model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
			G_model.add_output(name='output', input='D_softmax')

		elif fusion_type=='late':

			G_model.add_node(Dense(n_LSTM, 512, shared_weights_layer=D_model.nodes['fc0_RNN'], params_fixed=True), name='fc0_RNN' , input='RNN_scaled')
			G_model.add_node(Activation('relu'), name='fc0_RNN_relu', input='fc0_RNN')
			G_model.add_node(Dense(512, 256, shared_weights_layer=D_model.nodes['fc1_RNN'], params_fixed=True), name='fc1_RNN' , input='fc0_RNN_relu')
			G_model.add_node(Activation('relu'), name='fc1_RNN_relu', input='fc1_RNN')

			G_model.add_node(Dense(64*14*14, 1024, shared_weights_layer=D_model.nodes['fc0_prop'], params_fixed=True), name='fc0_prop' , input='proposed_flattened')
			G_model.add_node(Activation('relu'), name='fc0_prop_relu', input='fc0_prop')
			G_model.add_node(Dense(1024, 256, shared_weights_layer=D_model.nodes['fc1_prop'], params_fixed=True), name='fc1_prop' , input='fc0_prop_relu')
			G_model.add_node(Activation('relu'), name='fc1_prop_relu', input='fc1_prop')

			G_model.add_node(Dense(512, 128, shared_weights_layer=D_model.nodes['fc0_fus'], params_fixed=True), name='fc0_fus' , inputs=['fc1_prop_relu', 'fc1_RNN_relu'], merge_mode='concat', concat_axis=1)
			G_model.add_node(Activation('relu'), name='fc0_fus_relu', input='fc0_fus')
			G_model.add_node(Dense(128, 2, shared_weights_layer=D_model.nodes['fc1_fus'], params_fixed=True), name='fc1_fus' , input='fc0_fus_relu')
			G_model.add_node(Activation('softmax'), name='D_softmax', input='fc1_fus')
			G_model.add_output(name='output', input='D_softmax')


	elif G_model=='facegen_rotation_prednet_twolayer_G_to_D_features' and D_model=='facegen_rotation_prednet_twolayer_D_features':

		G_model = Graph()

		G_model.add_input(name='previous_frames', ndim=3)  # really features
		G_model.add_node(LSTM(64*14*14, 1024), name='RNN', input='previous_frames')

		G_model.add_input(name='random_input', ndim=2)

		G_model.add_node(Dense(1024+G_params['rand_size'], 2048), name='fc0', inputs=['RNN', 'random_input'], merge_mode='concat', concat_axis=1)
		G_model.add_node(Activation('relu'), name='fc0_relu', input='fc0')
		G_model.add_node(Dense(2048, 64*14*14), name='fc1', input='fc0_relu')
		G_model.add_node(Activation('relu'), name='fc1_relu', input='fc1')

		#G_model.add_node(Reshape(64, 14, 14), name='decoder_reshape', input='fc0')

		G_model.add_output(name='feature_output', input='fc1_relu')

		D_model = Graph()
		D_model.add_input(name='previous_frames', ndim=3)  # really features
		D_model.add_node(LSTM(64*14*14, 512), name='RNN', input='previous_frames')
		D_model.add_input(name='proposed_frames', ndim=2)
		D_model.add_node(Dense(64*14*14+512, 2048), name='fc0_D', inputs=['proposed_frames', 'RNN'], merge_mode='concat', concat_axis=1)
		D_model.add_node(Activation('relu'), name='fc0_relu_D', input='fc0_D')
		D_model.add_node(Dense(2048, 512), name='fc1_D', input='fc0_relu_D')
		D_model.add_node(Activation('relu'), name='fc1_relu_D', input='fc1_D')
		D_model.add_node(Dense(512, 2), name='fc2_D', input='fc1_relu_D')
		D_model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
		D_model.add_output(name='output', input='D_softmax')

		G_model.add_node()



	elif G_model=='bouncing_ball_rnn_twolayer_30x30_G_to_D' and D_model=='bouncing_ball_rnn_twolayer_30x30_D':

		G_model = Graph()

		G_model.add_input(name='previous_frames', ndim=3)  # input is (batch_size, n_time_steps, frame_size**2)

		# PREPARE INPUT
		G_model.add_node(CollapseTimesteps(2), name='collapse_time', input='previous_frames') # output: (batch_size*n_time_steps, frame_size**2)
		G_model.add_node(Reshape(1, 30, 30), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, frame_size, frame_size)
		# ENCODER
		G_model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', params_fixed=G_params['encoder_fixed']), name='conv0_0', input='frames')
		G_model.add_node(Activation('relu'), name='conv0_relu_0', input='conv0_0')
		G_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0', input='conv0_relu_0')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((32, 16, 16)), name='norm0', input='pool0_0')
			prev_node = 'norm0'
		else:
			prev_node = 'pool0_0'
		G_model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', params_fixed=G_params['encoder_fixed']), name='conv1_0', input=prev_node)
		G_model.add_node(Activation('relu'), name='conv1_relu_0', input='conv1_0')
		G_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0', input='conv1_relu_0')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((32, 7, 7)), name='norm1', input='pool1_0')
			prev_node = 'norm1'
		else:
			prev_node = 'pool1_0'

		# ENCODER FEATURES
		G_model.add_node(Flatten(), name='flatten_features_0', input=prev_node)
		G_model.add_node(ExpandTimesteps(ndim=3, batch_size=G_params['batch_size']), name='previous_features', input='flatten_features_0')  # output:  (128, n_time_steps, 32*7*7)
		# LSTM
		G_model.add_node(LSTM(32*7*7, G_params['n_LSTM'], return_sequences=False), name='RNN_0', input='previous_features') #used to be just 32*7*7 for nLSTM
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((G_params['n_LSTM'],)), name='norm2', input='RNN_0')
			prev_node = 'norm2'
		else:
			prev_node = 'RNN_0'

		if G_params['use_rand_input']:
			G_model.add_input(name='random_input', ndim=2)
			G_model.add_node(Dense(G_params['rand_size']+G_params['n_LSTM'], 32*7*7), name='fc0', inputs=[prev_node, 'random_input'], merge_mode='concat', concat_axis=1)
		else:
			G_model.add_node(Dense(G_params['n_LSTM'], 32*7*7), name='fc0', input=prev_node)
		G_model.add_node(Activation('relu'), name='fc0_relu', input='fc0')
		G_model.add_node(Reshape(32, 7, 7), name='pred_features_0', input='fc0_relu')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((32,7,7)), name='norm3', input='pred_features_0')
			prev_node = 'norm3'
		else:
			prev_node = 'pred_features_0'

		# DECODER
		G_model.add_node(UpSample2D((2,2)), name='unpool0_0', input=prev_node)  # (.., 32, 14, 14)
		G_model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full', params_fixed=G_params['decoder_fixed']), name='deconv0_0', input='unpool0_0')  # (.., 32, 16, 16)
		G_model.add_node(Activation('relu'), name='deconv0_relu_0', input='deconv0_0')
		G_model.add_node(UpSample2D((2,2)), name='unpool1_0', input='deconv0_relu_0') # (.., 32, 32, 32)
		G_model.add_node(Convolution2D(1, 32, 3, 3, border_mode='valid', params_fixed=G_params['decoder_fixed']), name='deconv1_0', input='unpool1_0')  # (.., 1, 30, 30)
		G_model.add_node(Activation('relu'), name='deconv1_relu_0', input='deconv1_0')
		G_model.add_node(Activation('satlu'), name='predicted_frame_0', input='deconv1_relu_0') # output:  (batch_size, 1, 30, 30)

		if G_params['use_pixel_output']:
			G_model.add_output(name='pixel_output', input='predicted_frame_0')

		G_model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', shared_weights_layer=G_model.nodes['conv0_0'], params_fixed=True), name='conv0_0_D', input='predicted_frame_0')
		G_model.add_node(Activation('relu'), name='conv0_relu_0_D', input='conv0_0_D')
		G_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0_D', input='conv0_relu_0_D')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((32, 16, 16)), name='norm0_D', input='pool0_0_D')
			prev_node = 'norm0_D'
		else:
			prev_node = 'pool0_0_D'
		G_model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', shared_weights_layer=G_model.nodes['conv1_0'], params_fixed=True), name='conv1_0_D', input=prev_node)
		G_model.add_node(Activation('relu'), name='conv1_relu_0_D', input='conv1_0_D')
		G_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0_D', input='conv1_relu_0_D')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((32, 7, 7)), name='norm1_D', input='pool1_0_D')
			prev_node = 'norm1_D'
		else:
			prev_node = 'pool1_0_D'
		G_model.add_node(Flatten(), name='flatten_features_0_D', input=prev_node)

		D_model = Graph()

		D_model.add_input(name='previous_frames', ndim=3)
		D_model.add_node(CollapseTimesteps(2), name='collapse_time', input='previous_frames') # output: (batch_size*n_time_steps, frame_size**2)
		D_model.add_node(Reshape(1, 30, 30), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, frame_size, frame_size)
		# ENCODER
		D_model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', shared_weights_layer=G_model.nodes['conv0_0'], params_fixed=D_params['encoder_fixed']), name='conv0_0', input='frames')
		D_model.add_node(Activation('relu'), name='conv0_relu_0', input='conv0_0')
		D_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0', input='conv0_relu_0')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((32, 16, 16)), name='norm0', input='pool0_0')
			prev_node = 'norm0'
		else:
			prev_node = 'pool0_0'
		D_model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', shared_weights_layer=G_model.nodes['conv1_0'], params_fixed=D_params['encoder_fixed']), name='conv1_0', input=prev_node)
		D_model.add_node(Activation('relu'), name='conv1_relu_0', input='conv1_0')
		D_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0', input='conv1_relu_0')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((32, 7, 7)), name='norm1', input='pool1_0')
			prev_node = 'norm1'
		else:
			prev_node = 'pool1_0'
		# ENCODER FEATURES
		D_model.add_node(Flatten(), name='flatten_features_0', input=prev_node)
		D_model.add_node(ExpandTimesteps(ndim=3, batch_size=D_params['batch_size']), name='previous_features', input='flatten_features_0')
		# Discriminator RNN
		if D_params['share_RNN']:
			RNN_shared_layer = G_model.nodes['RNN_0']
		else:
			RNN_shared_layer = None
		D_model.add_node(LSTM(32*7*7, D_params['n_LSTM'], return_sequences=False, shared_weights_layer=RNN_shared_layer), name='RNN_0', input='previous_features')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((D_params['n_LSTM'],)), name='norm2', input='RNN_0')
			rnn_node = 'norm2'
		else:
			rnn_node = 'RNN_0'

		D_model.add_input(name='proposed_frames', ndim=4)
		D_model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', shared_weights_layer=G_model.nodes['conv0_0'], params_fixed=True), name='conv0_0_D', input='proposed_frames')
		D_model.add_node(Activation('relu'), name='conv0_relu_0_D', input='conv0_0_D')
		D_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0_0_D', input='conv0_relu_0_D')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((32, 16, 16)), name='norm0_D', input='pool0_0_D')
			prev_node = 'norm0_D'
		else:
			prev_node = 'pool0_0_D'
		D_model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', shared_weights_layer=G_model.nodes['conv1_0'], params_fixed=True), name='conv1_0_D', input=prev_node)
		D_model.add_node(Activation('relu'), name='conv1_relu_0_D', input='conv1_0_D')
		D_model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1_0_D', input='conv1_relu_0_D')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((32, 7, 7)), name='norm1_D', input='pool1_0_D')
			prev_node = 'norm1_D'
		else:
			prev_node = 'pool1_0_D'
		D_model.add_node(Flatten(), name='proposed_flattened', input=prev_node)

		D_model.add_node(Dense(D_params['n_LSTM']+32*7*7, 512), name='fc0_D', inputs=[rnn_node, 'proposed_flattened'], merge_mode='concat', concat_axis=1)
		D_model.add_node(Activation('relu'), name='fc0_D_relu', input='fc0_D')
		if D_params['use_batch_norm']:
			D_model.add_node(BatchNormalization((512,)), name='norm3', input='fc0_D_relu')
			prev_node = 'norm3'
		else:
			prev_node = 'fc0_D_relu'
		D_model.add_node(Dense(512, 128), name='fc1_D', input=prev_node)
		D_model.add_node(Activation('relu'), name='fc1_D_relu', input='fc1_D')
		D_model.add_node(Dense(128, 2), name='fc2_D', input='fc1_D_relu')
		D_model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
		D_model.add_output(name='output', input='D_softmax')

		G_model.add_node(LSTM(32*7*7, D_params['n_LSTM'], return_sequences=False, shared_weights_layer=D_model.nodes['RNN_0'], params_fixed=True), name='RNN_0_D', input='previous_features')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((D_params['n_LSTM'],)), name='norm_LSTM_D', input='RNN_0_D')
			prev_node = 'norm_LSTM_D'
		else:
			prev_node = 'RNN_0_D'
		G_model.add_node(Dense(D_params['n_LSTM']+32*7*7, 512, shared_weights_layer=D_model.nodes['fc0_D'], params_fixed=True), name='fc0_D', inputs=[prev_node, 'flatten_features_0_D'], merge_mode='concat', concat_axis=1)
		G_model.add_node(Activation('relu'), name='fc0_D_relu', input='fc0_D')
		if G_params['use_batch_norm']:
			G_model.add_node(BatchNormalization((512,)), name='norm_fc_D', input='fc0_D_relu')
			prev_node = 'norm_fc_D'
		else:
			prev_node = 'fc0_D_relu'
		G_model.add_node(Dense(512, 128, shared_weights_layer=D_model.nodes['fc1_D'], params_fixed=True), name='fc1_D', input=prev_node)
		G_model.add_node(Activation('relu'), name='fc1_D_relu', input='fc1_D')
		G_model.add_node(Dense(128, 2, shared_weights_layer=D_model.nodes['fc2_D'], params_fixed=True), name='fc2_D', input='fc1_D_relu')
		G_model.add_node(Activation('softmax'), name='D_softmax', input='fc2_D')
		G_model.add_output(name='output', input='D_softmax')

	elif G_model=='facegen_refine_G' and D_model=='facegen_refine_D':

		G_model = Graph()

		G_model.add_input(name='mse_frames', ndim=4)
		G_model.add_input(name='real_frames', ndim=4)

		G_model.add_node(Convolution2D(64, 1, 5, 5, border_mode='valid'), name='conv0', input='mse_frames')
		G_model.add_node(Activation('relu'), name='relu0', input='conv0')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0', input='relu0')

		G_model.add_node(Convolution2D(64, 64, 5, 5, border_mode='valid'), name='conv1', input='pool0')
		G_model.add_node(Activation('relu'), name='relu1', input='conv1')
		G_model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool1', input='relu1')

		G_model.add_node(Flatten(), name='flatten', input='pool1')
		G_model.add_node(Dense(64*14*14, 1024), name='dense0', input='flatten')
		G_model.add_node(Activation('relu'), name='dense0_relu', input='dense0')
		G_model.add_node(Dense(1024, 64*14*14), name='dense1', input='dense0_relu')
		G_model.add_node(Activation('relu'), name='dense1_relu', input='dense1')
		G_model.add_node(Reshape(64, 14, 14), name='reshape', input='dense1_relu')

		G_model.add_node(UpSample2D((3,3)), name='unpool0', input='reshape')
		G_model.add_node(Convolution2D(64, 64, 7, 7, border_mode='full'), name='conv3', input='unpool0')
		G_model.add_node(Activation('relu'), name='relu3', input='conv3')

		G_model.add_node(UpSample2D((3,3)), name='unpool1', input='relu3')
		G_model.add_node(Convolution2D(1, 64, 7, 7, border_mode='full'), name='conv4', input='unpool1')
		G_model.add_node(Activation('relu'), name='relu4', input='conv4')
		G_model.add_node(Activation('satlu'), name='satlu0', input='relu4')

		G_model.add_output(name='pixel_output', input='satlu0')

		D_model = Graph()

		D_model.add_input(name='G_output', ndim=4)
		D_model.add_input(name='real_frames', ndim=4)

		m_map = {'G': ['_D_fake', '_D_real'], 'D': ['_fake', '_real']}

		for m in ['D', 'G']:
			if m=='G':
				model = G_model
				model.add_node(Convolution2D(32, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_fake'], params_fixed=True), name='conv0_D_fake', input='satlu0')
				model.add_node(Convolution2D(32, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_fake'], params_fixed=True), name='conv0_D_real', input='real_frames')
			else:
				model = D_model
				model.add_node(Convolution2D(32, 1, 5, 5, border_mode='valid'), name='conv0_fake', input='G_output')
				model.add_node(Convolution2D(32, 1, 5, 5, border_mode='valid', shared_weights_layer=D_model.nodes['conv0_fake'], params_fixed=True), name='conv0_real', input='real_frames')


			for s in m_map[m]:
				model.add_node(Activation('relu'), name='relu0'+s, input='conv0'+s)
				model.add_node(MaxPooling2D(poolsize=(3,3)), name='pool0'+s, input='relu0'+s)

				if s=='_fake':
					shared_layer = None
					p_fixed = False
				else:
					shared_layer = D_model.nodes['conv1_fake']
					p_fixed = True
				model.add_node(Convolution2D(64, 32, 5, 5, border_mode='valid', shared_weights_layer=shared_layer, params_fixed=p_fixed), name='conv1'+s, input='pool0'+s)
				model.add_node(Activation('relu'), name='relu1'+s, input='conv1'+s)
				model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1'+s, input='relu1'+s)
				if D_params['use_batch_norm']:
					model.add_node(BatchNormalization((64, 22, 22)), name='norm1'+s, input='pool1'+s)
					prev_node = 'norm1'+s
				else:
					prev_node = 'pool1'+s

				if s=='_fake':
					shared_layer = None
					p_fixed = False
				else:
					shared_layer = D_model.nodes['conv2_fake']
					p_fixed = True
				model.add_node(Convolution2D(128, 64, 5, 5, border_mode='valid', shared_weights_layer=shared_layer, params_fixed=p_fixed), name='conv2'+s, input=prev_node)
				model.add_node(Activation('relu'), name='relu2'+s, input='conv2'+s)
				model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool2'+s, input='relu2'+s)
				if D_params['use_batch_norm']:
					model.add_node(BatchNormalization((128, 9, 9)), name='norm2'+s, input='pool2'+s)
					prev_node = 'norm2'+s
				else:
					prev_node = 'pool2'+s

				model.add_node(Flatten(), name='flatten0'+s, input=prev_node)
				if s=='_fake':
					shared_layer = None
					p_fixed = False
				else:
					shared_layer = D_model.nodes['dense0_fake']
					p_fixed = True
				model.add_node(Dense(128*9*9, 1, shared_weights_layer=shared_layer, params_fixed=p_fixed), name='dense0'+s, input='flatten0'+s)

			#model.add_node(Subtract(), name='subtract', inputs=['dense0'+m_map[m][0], 'dense0'+m_map[m][1]], concat_axis=-1)
			if m=='G':
				model.add_node(Activation('softmax'), name='softmax', inputs=['dense0_D_real', 'dense0_D_fake'], concat_axis=1)
			else:
				model.add_node(Activation('softmax'), name='softmax', inputs=['dense0_real', 'dense0_fake'], concat_axis=1)
			model.add_output(name='output', input='softmax')



	return G_model, D_model




def add_mnist_cnn_feature_model0(model, tag, input_name, params_fixed=True, use_dropout=False):

	#given input of size (1, 28, 28) it returns size (32, 8, 8)

	model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full', params_fixed=params_fixed), name=tag+'_conv0', input=input_name)
	model.add_node(Activation('relu'), name=tag+'_conv_relu0', input=tag+'_conv0')
	model.add_node(MaxPooling2D(poolsize=(2,2)), name=tag+'_pool0', input=tag+'_conv_relu0')
	model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full', params_fixed=params_fixed), name=tag+'_conv1', input=tag+'_pool0')
	model.add_node(Activation('relu'), name=tag+'_conv_relu1', input=tag+'_conv1')
	model.add_node(MaxPooling2D(poolsize=(2,2)), name=tag+'_feature_output', input=tag+'_conv_relu1')
	if use_dropout:
		model.add_node(Dropout(0.25))

	return model


def add_mnist_cnn_deconv_model0(model, tag, input_name, params_fixed=True, use_dropout=True):

	#given input of size (1, 28, 28) it returns size (32, 8, 8)

	model.add_node(UnPooling2D(unpoolsize=(2,2)), name=tag+'_unpool0', input=input_name)
	model.add_node(Convolution2D(32, 32, 4, 4, border_mode='valid'), name=tag+'_deconv0', input=tag+'_unpool0')
	model.add_node(Activation('relu'), name=tag+'_deconv_relu0', input=tag+'_deconv0')
	if use_dropout:
		model.add_node(Dropout(0.25), name=tag+'_deconv_dropout0', input=tag+'_deconv_relu0')
		prev_node = tag+'_deconv_dropout0'
	else:
		prev_node = tag+'_deconv_relu0'
	model.add_node(UnPooling2D(unpoolsize=(2,2)), name=tag+'_unpool1', input=prev_node)
	model.add_node(Convolution2D(1, 32, 3, 3, border_mode='full'), name=tag+'_deconv1', input=tag+'_unpool1')
	model.add_node(Activation('relu'), name=tag+'_deconv_relu1', input=tag+'_deconv1')
	model.add_node(Activation('satlu'), name=tag+'_deconv_output', input=tag+'_deconv_relu1')

	return model


def add_mnist_prednet_D1(model, params, params_fixed=False):

	#input is previous_features and flatten_proposed

	if params['use_LSTM']:
		model.add_node(LSTM(32*8*8, params['n_RNN_units'], return_sequences=False, params_fixed=params_fixed), name='D_LSTM', input='previous_features')
	else:
		model.add_node(SimpleRNN(32*8*8, params['n_RNN_units'], return_sequences=False, params_fixed=params_fixed), name='D_LSTM', input='previous_features')
	model.add_node(Dropout(0.5), name='D_LSTM_dropout', input='D_LSTM')

	# concat LSTM encoding with generated encoding
	model.add_node(Dense(32*8*8+params['n_RNN_units'], 128, params_fixed=params_fixed), name='D_FC0', inputs=['D_LSTM_dropout', 'flatten_proposed'], concat_axis=1)
	#model.add_node(Dropout(0.25), name='D_FC0_dropout', input='D_FC0')
	model.add_node(Activation('relu'), name='D_FC0_relu', input='D_FC0')

	model.add_node(Dense(128, 64, params_fixed=params_fixed), name='D_FC1', input='D_FC0_relu')
	#model.add_node(Dropout(0.25), name='D_FC1_dropout', input='D_FC1')
	model.add_node(Activation('relu'), name='D_FC1_relu', input='D_FC1')

	model.add_node(Dense(64, 32, params_fixed=params_fixed), name='D_FC2', input='D_FC1_relu')
	#model.add_node(Dropout(0.25), name='D_FC2_dropout', input='D_FC2')
	model.add_node(Activation('relu'), name='D_FC2_relu', input='D_FC2')

	model.add_node(Dense(32, 2, params_fixed=params_fixed), name='D_FC3', input='D_FC2_relu')
	model.add_node(Activation('softmax'), name='D_softmax', input='D_FC3')
	model.add_output(name='output', input='D_softmax')

	return model

def mnist_prednet_G2(use_LSTM, n_RNN_units, rand_nx, fix_encoder=True):

	model = Graph()

	# input is (128, 2, 28*28*1)
	model.add_input(name='input_frames', ndim=3)
	#batch size must be 128, assumes 2 timesteps in input frame
	model.add_node(FullReshape(128*2, 28*28), name='collapse_time', input='input_frames')
	model.add_node(Reshape(1, 28, 28), name='reshape_frames', input='collapse_time')

	# encoder
	model = add_mnist_cnn_feature_model0(model, 'G', 'reshape_frames', params_fixed=fix_encoder, use_dropout=False)

	# format for LSTM
	model.add_node(Flatten(), name='G_flatten', input='G_feature_output')
	model.add_node(ExpandTimesteps(ndim=3, n_timesteps=2), name='previous_features', input='G_flatten')
	# previous_features will be (128, 2, 32*8*8)

	# LSTM
	if use_LSTM:
		model.add_node(LSTM(32*8*8, n_RNN_units, return_sequences=False), name='G_RNN', input='previous_features')
	else:
		model.add_node(SimpleRNN(32*8*8, n_RNN_units, return_sequences=False), name='G_RNN', input='previous_features')
	model.add_node(Dropout(0.25), name='G_RNN_output', input='G_RNN')

	# transform random_input
	model.add_input(name='random_input', ndim=4)
	model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv0', input='random_input')
	model.add_node(Activation('relu'), name='randconv_relu0', input='randconv0')
	model.add_node(Convolution2D(1, 1, 3, 3, border_mode='valid'), name='randconv1', input='randconv_relu0')
	model.add_node(Activation('relu'), name='randconv_relu1', input='randconv1')
	model.add_node(Flatten(), name='rand_vector', input='randconv_relu1')

	# Fully connected layer and concat LSTM with random
	model.add_node(Dense(n_RNN_units+(rand_nx-4)**2, 1024), name='G_FC0', inputs=['G_RNN_output', 'rand_vector'], concat_axis=1)
	model.add_node(Activation('relu'), name='G_FC_relu0', input='G_FC0')
	model.add_node(Dropout(0.25), name='G_FC_dropout0', input='G_FC_relu0')
	model.add_node(Dense(1024, 2048), name='G_FC1', input='G_FC_dropout0')
	model.add_node(Activation('relu'), name='G_FC_relu1', input='G_FC1')
	model.add_node(Dropout(0.25), name='G_FC_dropout1', input='G_FC_relu1')
	model.add_node(Reshape(32, 8, 8), name='G_proposed_features', input='G_FC_dropout1')
	# output is (128, 32, 8, 8)


	return model


def add_mnist_prednet_D2(model, RNN_input_name, proposed_input_name, use_LSTM, n_RNN_units, params_fixed=False, use_dropout=True):

	# RNN input is (128, 2, 32*8*8)
	if use_LSTM:
		model.add_node(LSTM(32*8*8, n_RNN_units, return_sequences=False, params_fixed=params_fixed), name='D_RNN', input=RNN_input_name)
	else:
		model.add_node(SimpleRNN(32*8*8, n_RNN_units, return_sequences=False, params_fixed=params_fixed), name='D_RNN', input=RNN_input_name)
	model.add_node(Dropout(0.25), name='D_RNN_output', input='D_RNN')

	# flatten proposed features
	model.add_node(Flatten(), name='proposed_flattened', input=proposed_input_name)

	# concat LSTM encoding with generated encoding
	model.add_node(Dense(32*8*8+n_RNN_units, 128, params_fixed=params_fixed), name='D_FC0', inputs=['D_RNN_output', 'proposed_flattened'], concat_axis=1)
	if use_dropout:
		model.add_node(Dropout(0.25), name='D_FC0_dropout', input='D_FC0')
		n = 'D_FC0_dropout'
	else:
		n = 'D_FC0'
	model.add_node(Activation('relu'), name='D_FC0_relu', input=n)

	model.add_node(Dense(128, 32, params_fixed=params_fixed), name='D_FC1', input='D_FC0_relu')
	if use_dropout:
		model.add_node(Dropout(0.25), name='D_FC1_dropout', input='D_FC1')
		n = 'D_FC1_dropout'
	else:
		n = 'D_FC1'
	model.add_node(Activation('relu'), name='D_FC1_relu', input=n)

	model.add_node(Dense(32, 2, params_fixed=params_fixed), name='D_FC2', input='D_FC1_relu')
	model.add_node(Activation('softmax'), name='D_output', input='D_FC2')

	return model


def load_model(model_name, weights_file = None):



	# if weights_file is None:
	# 	if model_name == 'mnist_cnn0':
	# 		weights_file = '/home/bill/Projects/Predictive_Networks/models/mnist_cnn0_weights0.hdf5'

	if model_name=='vgg_16':
		model = load_vgg16_model()
	if model_name=='vgg_16_test':
		model = load_vgg16_test_model()
	else:
		model = initialize_model(model_name)
		weights_file = '/home/bill/Projects/Predictive_Networks/models/'+model_name+'_weights0.hdf5'

		model.load_weights(weights_file)

	return model


def load_vgg16_model_old():

	model = initialize_model('vgg_16')

	import scipy.io as spio
	data = spio.loadmat('/home/bill/Data/matconvnet_models/imagenet-vgg-verydeep-16.mat')

	for i in range(len(data['layers'][0])):
		if len(data['layers'][0][i][0][0])==5:
			l_name = data['layers'][0][i][0][0][3][0]
			w = data['layers'][0][i][0][0][0][0][0]
			b = data['layers'][0][i][0][0][0][0][1]
			b = b.reshape((b.shape[1],))

			if 'fc' in l_name:
				w2 = np.zeros((w.shape[0]*w.shape[1]*w.shape[2], w.shape[3])).astype(np.float32)
				for i in range(w.shape[3]):
					w2[:,i] = np.reshape(w[:,:,:,i], (w.shape[0]*w.shape[1]*w.shape[2]))
				w = w2
			else:
				w = np.transpose(w, (3, 2, 0, 1))

			model.nodes[l_name].set_weights((w,b))

	return model


def load_vgg16_test_model():

	model = initialize_model('vgg_16_test')

	import scipy.io as spio
	data = spio.loadmat('/home/bill/Data/matconvnet_models/imagenet-vgg-verydeep-16.mat')

	pdb.set_trace()
	for i in range(2):
		if len(data['layers'][0][i][0][0])==5:
			l_name = data['layers'][0][i][0][0][3][0]
			w = data['layers'][0][i][0][0][0][0][0]
			b = data['layers'][0][i][0][0][0][0][1]
			b = b.reshape((b.shape[1],))

			if 'fc' in l_name:
				w2 = np.zeros((w.shape[0]*w.shape[1]*w.shape[2], w.shape[3])).astype(np.float32)
				for i in range(w.shape[3]):
					w2[:,i] = np.reshape(w[:,:,:,i], (w.shape[0]*w.shape[1]*w.shape[2]))
				w = w2
			else:
				w = np.transpose(w, (3, 2, 0, 1))

			model.nodes[l_name].set_weights((w,b))

	return model


def load_vgg16_test_model2():

	model = initialize_model('vgg_16_test')

	import scipy.io as spio
	data = spio.loadmat('/home/bill/Data/matconvnet_models/imagenet-vgg-verydeep-16_processed.mat')

	w = data['weights'][0][0]
	b = data['biases'][0][0].flatten()
	model.nodes['conv1_1'].set_weights((w,b))

	return model

def load_vgg16_model():

	model = initialize_model('vgg_16')

	import scipy.io as spio
	data = spio.loadmat('/home/bill/Data/matconvnet_models/imagenet-vgg-verydeep-16_processed.mat')

	layers = []
	for k in data['keys']:
		layers.append(str(k[0][0]))

	for idx,key in enumerate(layers):
		w = data['weights'][idx][0]
		b = data['biases'][idx][0].flatten()
		model.nodes[key].set_weights((w,b))

	return model


def plot_model(model):

	G = nx.DiGraph()
	for node in model.node_config:
		node_name = node['name']
		if len(node['inputs'])==0:
			inputs = [node['input']]
		else:
			inputs = node['inputs']
		for i in inputs:
			G.add_edge(i, node_name)

	nx.draw_graphviz(G, with_labels=True)
	plt.show()


	pdb.set_trace()


if __name__=='__main__':
	try:
		model = load_vgg16_model()

	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
