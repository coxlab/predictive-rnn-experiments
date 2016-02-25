import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model

sys.path.append('/home/bill/Libraries/keras/')
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *

from copy import deepcopy
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import six.moves.cPickle

if os.path.isdir('/home/bill/'):
	sys.path.append('/home/bill/Dropbox/Cox_Lab/General_Scripts/')
	import general_python_functions as gp
	sys.path.append('/home/bill/Libraries/keras/')
	from keras.datasets import mnist
	from keras.utils import np_utils



deconv_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/deconv_runs/'
prediction_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/prediction_runs/'

# layer_num is 0 indexed
def extract_features(model, layer_num, X):

	model2 = deepcopy(model)
	model2.layers = model2.layers[:layer_num+1]
	model2._predict = theano.function([model2.layers[0].get_input(train=False)], model2.layers[layer_num].get_output(train=False),
            allow_input_downcast=True)
	#model2._predict = theano.function([model2.get_input(train=False)], model.layers[layer_num].get_output(train=False),
    #        allow_input_downcast=True)
	#model2.compile(loss='categorical_crossentropy', optimizer='adadelta') # the actual loss and optimizer won't be used
	features = model2.predict(X)
	#pdb.set_trace()

	#import theano
	#get_features = theano.function([model.layers[0].get_input(train=False)], model.layers[layer_num].get_output(train=False), allow_input_downcast=True)
	#batch_size = 128
	#features = get_features(X)
	return features

def extract_features_graph(model, layer_names, X):

	model2 = deepcopy(model)

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


def run_extract_features(layer_num):

	model_name = 'mnist_cnn0'
	model = load_model(model_name)

	(X_train, y_train), (X_test, y_test) = load_mnist()

	features_train = extract_features(model, layer_num, X_train)
	features_test = extract_features(model, layer_num, X_test)

	return features_train, features_test


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

def train_forward_net(model_name):

	batch_size = 128
	nb_epoch = 5

	model = initialize_model(model_name)

	(X_train, y_train), (X_test, y_test) = load_mnist()
	X_train = X_train[:30000]
	y_train = y_train[:30000]

	model.compile(loss='categorical_crossentropy', optimizer='adadelta')

	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	model.save_weights('/home/bill/Projects/Predictive_Networks/models/'+model_name+'_weights0.hdf5')

def flatten_features(X):
	size = np.prod(X.shape)/X.shape[0]
	return X.reshape( (X.shape[0], size) )

# for graphs
def train_forward_net2(model_name):

	batch_size = 128
	nb_epoch = 20

	model = initialize_model(model_name)

	(X_train, y_train), (_, _) = load_mnist()
	X_test = X_train[50000:]
	y_test = y_train[50000:]
	X_train = X_train[:50000]
	y_train = y_train[:50000]

	losses = {'classification_output': 'categorical_crossentropy'}
	model.compile(optimizer='rmsprop', loss=losses)

	model.fit({'input': X_train, 'classification_output': y_train}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
	y_hat = model.predict({'input': X_test})
	acc = np.mean(np.argmax(y_hat['classification_output'], axis=-1) == np.argmax(y_test, axis=-1))
	print('Test accuracy:', acc)

	model.save_weights('/home/bill/Projects/Predictive_Networks/models/'+model_name+'_weights0.hdf5')


def train_full_prediction_net():

	pred_model_name = 'mnist_cnn2_predictor0'
	forward_model_name = 'mnist_cnn2'
	forward_feature_layers = ['dropout0', 'dropout1']
	forward_feature_shapes = [14, 8]
	output_size = 26
	input_layer = 'flatten0'
	n_frames_in = 10
	n_frames_predict = 1
	batch_size = 128
	nb_epoch = 500

	forward_model = load_model(forward_model_name)

	print 'Loading Data'
	X = {}
	X['test'] = load_mnist_clips('test')

	X['train'] = X['test'][:9900]
	X['test'] = X['test'][9900:]


	for t in ['train', 'test']:
		X_frames_in = X[t][:,:n_frames_in].reshape( (X[t].shape[0]*n_frames_in,) + X[t].shape[2:] )
		print 'Calculating Input Features for '+t
		input_features = extract_features_graph(forward_model, input_layer, {'input': X_frames_in})[input_layer]
		n_clips = X[t].shape[0]
		input_features = np.reshape(input_features, (n_clips, n_frames_in, np.prod(input_features.shape[1:])))
		X_frames_predict = X[t][:,n_frames_in:n_frames_in+n_frames_predict].reshape( (X[t].shape[0]*n_frames_predict,) + X[t].shape[2:] )


		if t=='train':
			print 'Calculating Layer Features for '+t
			predict_features = extract_features_graph(forward_model, forward_feature_layers, {'input': X_frames_predict})
			for i,f in enumerate(forward_feature_layers):
				if predict_features[f].shape[-1] != forward_feature_shapes[i]:
					predict_features[f] = resize_image_stack(predict_features[f], forward_feature_shapes[i])
				predict_features[f] = flatten_features(predict_features[f])

			if X_frames_predict.shape[-1] != output_size:
				X_frames_predict = resize_image_stack(X_frames_predict, output_size)

			model = initialize_model(pred_model_name)
			model.compile('rmsprop', {'decoder_output': 'mse', 'deconv0_output': 'mse', 'prediction_output': 'mse'})
			model.fit({'input': input_features, 'decoder_output': predict_features['dropout1'], 'deconv0_output': predict_features['dropout0'], 'prediction_output': flatten_features(X_frames_predict)}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
			#model.fit({'input': input_features, 'decoder_output': np.random.uniform(size=(9900, 32*8*8)), 'deconv0_output': np.random.uniform(size=(9900, 14*14*32)), 'prediction_output': np.random.uniform(size=(9900, 26*26*1))}, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
		else:
			if X_frames_predict.shape[-1] != output_size:
				X_frames_predict = resize_image_stack(X_frames_predict, output_size)
			y_hat = model.predict({'input': input_features})['prediction_output'].reshape( (input_features.shape[0], 1, output_size, output_size) )

	run_num = gp.get_next_run_num(prediction_dir)
	save_dir = prediction_dir + 'run_' + str(run_num) + '/'
	os.mkdir(save_dir)

	f = open(save_dir + 'results.pkl', 'w')
	pickle.dump([y_hat[:40], X_frames_predict[:40]], f)
	f.close()

	plot_prediction_results(run_num)



def train_deconv_net():

	batch_size = 128
	nb_epoch = 5
	features_layer_num = 5
	deconv_num = 3
	(X_train, y_train), (X_test, y_test) = load_mnist()
	X_train = X_train[30000:]
	y_train = y_train[30000:]

	features_train, features_test = run_extract_features(layer_num=features_layer_num)

	X = {}
	(X['train'], y_train), (X['test'], y_test) = load_mnist()

	X_resampled = {}
	for t in ['train', 'test']:
		X_resampled[t] = np.zeros((X[t].shape[0], X[t].shape[1], 24, 24))
		for i in range(X[t].shape[0]):
			im = scipy.ndimage.zoom(X[t][i,0,:,:], 24.0/float(X[t].shape[2]))
			im[im<0.01] = 0
			X_resampled[t][i,0,:,:] = im


	model = initialize_model('mnist_cnn_deconv0_'+str(deconv_num))

	model.compile(loss='mean_squared_error', optimizer='adadelta')

	model.fit(features_train, X_resampled['train'], batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(features_test, X_resampled['test']))
	y_hat = model.predict(features_test)

	run_num = gp.get_next_run_num(deconv_dir)
	save_dir = deconv_dir + 'run_' + str(run_num) + '/'
	os.mkdir(save_dir)
	f = open(save_dir + 'results.pkl', 'w')
	pickle.dump([y_hat[:10], X_resampled['test'][:10]], f)
	f.close()

	plot_results(run_num)

def resize_image_stack(im_stack, size, threshold = 0.01):

	imstack_resampled = np.zeros((im_stack.shape[0], im_stack.shape[1], size, size))
	for i in range(im_stack.shape[0]):
		for j in range(im_stack.shape[1]):
			im = scipy.ndimage.zoom(im_stack[i,j], float(size)/im_stack.shape[2])
			im[im<threshold] = 0
			imstack_resampled[i,j]= im

	return imstack_resampled

def train_prediction_net():

	batch_size = 128
	nb_epoch = 10
	model_name = 'mnist_rotation_predictor0_4'
	threshold = 0.01
	n_frames_in = 10
	n_frames_predict = 1
	feature_model_name = 'mnist_cnn0'
	feature_layer_num = 8

	feature_model = load_model(feature_model_name)

	X = {}
	y = {}

	t0 = time.time()
	print 'Loading Data...'
	for t in ['test']:
		X[t] = load_mnist_clips(t)

	X['train'] = X['test'][:9000]
	X['test'] = X['test'][9000:]
	print 'Done Loading data, elapsed: ' +str(np.round(time.time()-t0)) + ' seconds'

	run_num = gp.get_next_run_num(prediction_dir)
	save_dir = prediction_dir + 'run_' + str(run_num) + '/'
	os.mkdir(save_dir)

	for t in ['train','test']:
		# X[t] is (n_ims,n_frames, 1, nx, ny)
		n_clips = X[t].shape[0]
		#clip_frames_predict = X[t][:,n_frames_in:]
		clip_frames_predict = X[t][:,n_frames_in:n_frames_in+n_frames_predict]
		clip_frames_features = X[t][:,:n_frames_in]
		clip_frames_predict = np.reshape(clip_frames_predict, (X[t].shape[0]*clip_frames_predict.shape[1], X[t].shape[2], X[t].shape[3], X[t].shape[4]))
		clip_frames_features = np.reshape(clip_frames_features, (X[t].shape[0]*clip_frames_features.shape[1], X[t].shape[2], X[t].shape[3], X[t].shape[4]))

		#del X[t]
		# features has to be like (n_ims, 1, nx, ny)
		print 'Generating features for '+t
		features = extract_features(feature_model, layer_num=feature_layer_num, X=clip_frames_features)
		features = features.reshape((n_clips, n_frames_in, np.prod(features.shape[1:])))

		# pdb.set_trace()
		output_size = 24
		clips_frames_predict_resampled = np.zeros((clip_frames_predict.shape[0], 1, output_size, output_size))
		for i in range(clip_frames_predict.shape[0]):
			im = scipy.ndimage.zoom(clip_frames_predict[i,0], float(output_size)/clip_frames_predict.shape[2])
			im[im<threshold] = 0
			clips_frames_predict_resampled[i,0]= im

		if t=='train':
			model = initialize_model(model_name)
			model.compile(loss='mean_squared_error', optimizer='adadelta')
			model.fit(features, clips_frames_predict_resampled, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1)
		else:
			Y_hat = model.predict(features)


	f = open(save_dir + 'results.pkl', 'w')
	pickle.dump([Y_hat[:20], clips_frames_predict_resampled[:20]], f)
	f.close()

	plot_prediction_results(run_num)


def load_mnist_clips(tag):

	f = open('/home/bill/Data/MNIST/Rotation_Clips/mnist_rotations_15-frames-per-clip_'+tag+'.pkl','rb')
	X = six.moves.cPickle.load(f)

	return X


def get_feature_size_by_layer(layer_num):

	if layer_num==2:
		return (28, 28)


def run_generate_clips():

	n_frames = 15

	X = {}
	(X['train'], _), (X['test'], _) = load_mnist()
	X['train'] = X['train'][:30000]
	X['test'] = X['test'][:30000]

	for t in ['test']:
		clip_frames = generate_clip_set(X[t], n_frames)
		f = '/home/bill/Data/MNIST/Rotation_Clips/mnist_rotations_'+str(n_frames)+'-frames-per-clip_'+t+'.pkl'
		fid = open(f,'wb')
		six.moves.cPickle.dump(clip_frames, fid)

	# d = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/example_clips/'
	# for i in range(10):
	# 	for j in range(15):
	# 		plt.imshow(clip_frames[i,j,0])
	# 		plt.savefig(d+'clip_'+str(i)+'_'+str(j)+'.jpg')


def generate_clip_set(orig_ims, n_frames):

	n_ims = orig_ims.shape[0]
	nx = orig_ims.shape[2]

	clip_frames = np.zeros((n_ims, n_frames, 1, 28, 28))

	for i in range(n_ims):

		center_x = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
		center_y = np.random.randint(int(np.round(3*float(nx)/8)), 1+int(np.round(float(5*nx)/8)))
		theta0 = np.random.uniform(-np.pi/8, np.pi/8)
		angular_speed = np.random.uniform(-np.pi/10, np.pi/10)
		clip_frames[i,:,0] = create_rotation_clip(orig_ims[i,0], center_x, center_y, theta0, angular_speed, n_frames)

		if i%100==0:
			print 'Done clip '+str(i+1)+'/'+str(n_ims)

	return clip_frames


def train_class_and_pred_net():

	model_name = 'mnist_rotation_predandclass0'

	model = initialize_model(model_name)

	losses = {'classification_output': 'categorical_crossentropy', 'decoder_output': 'mse', 'deconv0_output': 'mse', 'prediction_output': 'mse'}
	model.compile(optimizer='rmsprop', loss=losses)

	model.fit(features_train, X_resampled['train'], batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(features_test, X_resampled['test']))
	predictions = model.predict(features_test)


def plot_results(run_num):

	save_dir = deconv_dir + 'run_' + str(run_num) + '/'
	f = open(save_dir + 'results.pkl', 'r')
	y_hat, y = pickle.load(f)
	f.close()

	pic_dir = save_dir + 'images/'
	if not os.path.isdir(pic_dir):
		os.mkdir(pic_dir)

	plt.figure()
	for i in range(y_hat.shape[0]):
		plt.subplot(1, 2, 1)
		plt.imshow(y[i,0])
		plt.xlabel('Original')

		plt.subplot(1, 2, 2)
		plt.imshow(y_hat[i,0])
		plt.xlabel('Reconstructed')

		plt.title('run_'+str(run_num))
		plt.savefig(pic_dir + 'im' + str(i) + '.jpg')

def plot_prediction_results(run_num):

	save_dir = prediction_dir + 'run_' + str(run_num) + '/'
	f = open(save_dir + 'results.pkl', 'r')
	y_hat, y = pickle.load(f)
	f.close()

	pic_dir = save_dir + 'images/'
	if not os.path.isdir(pic_dir):
		os.mkdir(pic_dir)

	n_plot = 20
	plt.figure()
	for i in range(n_plot):
		plt.subplot(1, 2, 1)
		plt.imshow(y[i,0])
		plt.xlabel('Original')

		plt.subplot(1, 2, 2)
		plt.imshow(y_hat[i,0])
		plt.xlabel('Reconstructed')

		plt.title('run_'+str(run_num))
		plt.savefig(pic_dir + 'prediction_im' + str(i) + '.jpg')


def plot_original_sequences():

	X = load_mnist_clips('test')
	start_num = 9900
	X = X[start_num:]
	n_plot = 20
	plt.figure()
	for i in range(n_plot):
		for j in range(15):
			plt.imshow(X[i,j,0])
			plt.title('Rotated Image Test Clip'+str(start_num+i)+' Frame'+str(j))
			plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/sequence_plots/clip_'+str(start_num+i)+'_frame_'+str(j)+'.jpg')



def LSTM_test():

	model = Graph()
	model.add_input(name='input', ndim=3)
	model.add_node(LSTM(100, 200, return_sequences=False), input='input', name='LSTM')
	#model.add_node(Flatten(), input='LSTM', name='flatten')
	model.add_output(name='output', input='LSTM')

	model.compile('rmsprop', {'output': 'mse'})
	model.fit({'input': np.random.uniform(size=(1000, 10, 100)), 'output': np.random.uniform(size=(1000, 200))})

def rotate_image(im, theta, center_x, center_y, threshold = 0.01):

	im_rotated = np.zeros_like(im)

	# should just pad the image and then crop

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

	#plt.figure()
	#plt.imshow(im_rotated)
	#plt.savefig('/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/rotated_im.jpg')

	return im_rotated

def create_rotation_clip(orig_im, center_x, center_y, theta0, angular_speed, n_frames):

	clip = np.zeros((n_frames, orig_im.shape[0], orig_im.shape[1]))

	for i in range(n_frames):
		clip[i] = rotate_image(orig_im, theta0+i*angular_speed, center_x, center_y)

	return clip


if __name__ == '__main__':
	try:
		#train_deconv_net()
		#plot_results(2)
		train_full_prediction_net()
		#plot_prediction_results(4)
		#run_generate_clips()
		#train_forward_net2('mnist_cnn2')
		#LSTM_test()

		#plot_original_sequences()



	except:
		ty, value, tb = sys.exc_info()
		traceback.print_exc()
		pdb.post_mortem(tb)
