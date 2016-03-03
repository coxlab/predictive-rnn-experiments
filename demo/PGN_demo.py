
keras_dir = '/home/bill/Libraries/keras/'
save_dir = '/home/bill/tmp/'
data_file_train = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_training_set.hkl'
data_file_val = '/home/bill/Data/Bouncing_Balls/clip_set4/bouncing_balls_validation_set.hkl'

import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl
import sys
sys.path.append(keras_dir)
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.layers.normalization import *



nt_input = 10   # will input 10 frames into the model and predict the 11th
n_epochs = 20
batch_size = 10


def get_model():

    model = Graph()

    model.add_input(name='input_frames', ndim=3)  # input is (batch_size, n_time_steps, frame_size**2)

    # PREPARE INPUT
    model.add_node(CollapseTimesteps(2), name='collapse_time', input='input_frames') # output: (batch_size*n_time_steps, frame_size**2)
    model.add_node(Reshape(1, 30, 30), name='frames', input='collapse_time') # (batch_size*n_time_steps, 1, frame_size, frame_size)

    # ENCODER
    model.add_node(Convolution2D(32, 1, 3, 3, border_mode='full'), name='conv0', input='frames')
    model.add_node(Activation('relu'), name='conv0_relu', input='conv0')
    model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool0', input='conv0_relu')
    model.add_node(Convolution2D(32, 32, 3, 3, border_mode='valid'), name='conv1', input='pool0')
    model.add_node(Activation('relu'), name='conv1_relu', input='conv1')
    model.add_node(MaxPooling2D(poolsize=(2,2)), name='pool1', input='conv1_relu')

    # LSTM
    model.add_node(Flatten(), name='flatten_features', input='pool1')
    model.add_node(ExpandTimesteps(ndim=3, batch_size=batch_size), name='previous_features', input='flatten_features')  # output:  (128, n_time_steps, 32*7*7)
    model.add_node(LSTM(32*7*7, 32*7*7, return_sequences=False), name='RNN', input='previous_features')
    model.add_node(Reshape(32, 7, 7), name='pred_features', input='RNN')

    # DECODER
    model.add_node(UpSample2D((2,2)), name='unpool0', input='pred_features')  # (.., 32, 14, 14)
    model.add_node(Convolution2D(32, 32, 3, 3, border_mode='full'), name='deconv0', input='unpool0')  # (.., 32, 16, 16)
    model.add_node(Activation('relu'), name='deconv0_relu', input='deconv0')
    model.add_node(UpSample2D((2,2)), name='unpool1', input='deconv0_relu') # (.., 32, 32, 32)
    model.add_node(Convolution2D(1, 32, 3, 3, border_mode='valid'), name='deconv1', input='unpool1')  # (.., 1, 30, 30)
    model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')
    model.add_node(Activation('satlu'), name='predicted_frame', input='deconv1_relu') # output:  (batch_size, 1, 30, 30)

    model.add_output(name='output', input='predicted_frame')

    return model



def train():

    model = get_model()
    print 'Compiling Model...'
    model.compile(optimizer='rmsprop', loss={'output': 'mse'})

    # load and prepare training data
    print 'Loading Data...'
    X = hkl.load(open(data_file_train))
    X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))

    for epoch in range(n_epochs):
        # for each epoch, will select different time sequences
        start_t = np.random.randint(0, X_flat.shape[1]-nt_input)
        data = {'input_frames': X_flat[:,start_t:start_t+nt_input]}
        data['output'] = X[:,start_t+nt_input].reshape((X.shape[0], 1, X.shape[2], X.shape[3]))
        print "Epoch: "+str(epoch)
        model.fit(data, batch_size=batch_size, nb_epoch=1, verbose=1)

    return model



def evaluate(model):

    n_plot = 5  # how many examples to plot
    nt_predict = 8   # how many timesteps will predict for

    # load and prepare validation data
    X = hkl.load(open(data_file_val))
    X_flat = X.reshape((X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))

    predictions = np.zeros((X.shape[0], nt_predict, X.shape[-2], X.shape[-1]))

    # make predictions for each time step
    for t in range(nt_predict):
        data = {'input_frames': X_flat[:,:nt_input+t]}
        yhat = model.predict(data, batch_size=batch_size)
        predictions[:,t] = yhat['output'].reshape((X.shape[0], X.shape[-2], X.shape[-1]))

    # plot predictions
    plt.figure()
    for i in range(n_plot):
        for t in range(nt_predict):
            plt.subplot(2, nt_predict, t+1)
            plt.imshow(X[i, nt_input+t], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            if t==0:
                plt.ylabel('Actual')

            plt.subplot(2, nt_predict, t+nt_predict+1)
            plt.imshow(predictions[i, t], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.gca().axes.get_xaxis().set_ticks([])
            plt.gca().axes.get_yaxis().set_ticks([])
            if t==0:
                plt.ylabel('Predicted')

        plt.savefig(save_dir + 'predictions_clip_' + str(i) + '.jpg')



def run():

    model = train()
    evaluate(model)



if __name__=='__main__':
    run()
