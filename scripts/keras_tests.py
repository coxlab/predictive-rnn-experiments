import theano as T
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model, load_vgg16_model, load_vgg16_test_model
from prednet import *

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
	from keras.models import *
	from keras.optimizers import *
	sys.path.append('/home/bill/Libraries/keras/caffe/')
	from keras.layers.core import *
	from keras.layers.convolutional import *
	from keras.layers.recurrent import *

import pydot
np.random.seed(1337)

model = Graph()

model.add_input(name='input', ndim=4)

if True:
	model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0', input='input')
	model.add_node(Activation('relu'), name='relu0', input='conv0')
	model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='relu0')
	model.add_node(Dropout(0.25), name='dropout0', input='pool0')

	model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full', params_fixed=True), name='conv1', input='dropout0')
	model.add_node(Activation('relu'), name='relu1', input='conv1')
	model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='relu1')
	model.add_node(Dropout(0.25), name='dropout1', input='pool1')

	model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid', shared_weights_layer=model.nodes['conv1']), name='conv2', input='dropout1')
	model.add_node(Activation('relu'), name='relu2', input='conv2')
	model.add_node(Dropout(0.25), name='dropout2', input='relu2')

	# flatten stacks
	model.add_node(Flatten(), name='flatten0', input='dropout2')

	# classification loss
	model.add_node(Dense(32*6*6, 128), name='classification_dense0', input='flatten0')
	model.add_node(Activation('relu'), name='classification_relu0', input='classification_dense0')
	model.add_node(Dropout(0.5), name='classification_dropout0', input='classification_relu0')

	model.add_node(Dense(128, 10), name='classification_dense1', input='classification_dropout0')
	model.add_node(Activation('softmax'), name='classification_softmax0', input='classification_dense1')
	model.add_output(name='classification_output', input='classification_softmax0')

	(X, y), _ = load_mnist()


	model.compile(loss={'classification_output': 'categorical_crossentropy'}, optimizer='adadelta')
	#T.printing.pydotprint(model._train, outfile='/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_graph.ps')

	w1 = model.nodes['conv1'].get_weights()
	w2 = model.nodes['conv2'].get_weights()
	print 'initialization:'
	pdb.set_trace()
	#print 'w1==w2: '+str(w1[0]==w2[0] and w1[1]==w2[1])
	print
	model.fit({'input': X, 'classification_output': y}, batch_size=128, nb_epoch=3, verbose=1)
	w1 = model.nodes['conv1'].get_weights()
	w2 = model.nodes['conv2'].get_weights()
	print 'after fit:'
	pdb.set_trace()
	#print 'w1==w2: '+str(w1[0]==w2[0] and w1[1]==w2[1])
	print

else:
	model.add_node(Convolution2D(1, 1, 3, 3, border_mode='full'), name='conv0', input='input')
	W = model.nodes['conv0'].W
	b = model.nodes['conv0'].b
	model.add_node(Convolution2D(1, 1, 3, 3, border_mode='full', W=W, b=b), name='conv1', input='conv0')
	model.add_output(name='output', input='conv1')
	model.compile(loss={'output': 'mse'}, optimizer='SGD')
	T.printing.pydotprint(model._train, outfile='/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_graph.ps')
