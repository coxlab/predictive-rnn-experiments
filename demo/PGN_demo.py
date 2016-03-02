
sys.path.append('/home/bill/Libraries/keras/')
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.layers.normalization import *



model = Graph()

model.add_input(name='input_frames', batch_input_shape=(batch_size, n_time_steps, frame_size**2))  # input is (batch_size, n_time_steps, frame_size**2)

# PREPARE INPUT
model.add_node(FullReshape, name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, frame_size**2)
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
