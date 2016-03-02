import theano
import pdb, sys, traceback, os, pickle, time
from keras_models import load_model, initialize_model, load_vgg16_model, load_vgg16_test_model, plot_model, initialize_GAN_models
from prednet import load_mnist
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
from facegen_rotations import facegen_construct, morph_face, facegen_render
import cv2.cv as cv

from prednet import get_block_num_timesteps

def_dir = os.path.expanduser('~/default_dir')
sys.path.insert(0,def_dir)
from basic_fxns import *

cname = get_computer_name()

sys.path.append(get_scripts_dir() +'General_Scripts/')

import general_python_functions as gp
sys.path.append('/home/bill/Libraries/keras/')
from keras.models import *
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *

from prednet_feature_analysis import run_extract_features, get_param_values, run_fit_features
from prednet2 import initialize_weights, append_predict

# is_GAN = False
#
# if is_GAN:
#     base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_307/'
# else:
#     base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/'

pretty_names = {'pan_angular_speeds': 'rotation speed', 'pan_angles': 'angle', 'pan_initial_angles': 'initial angle'}
for i in range(6):
    pretty_names['pca_'+str(i)] = 'pca'+str(i)

def create_features(is_GAN):

    po = {}
    po['is_GAN'] = is_GAN
    po['is_autoencoder'] = False
    if is_GAN:
        base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_635/'
        po['model_name'] = 'facegen_rotation_prednet_twolayer_G_to_D'
        po['model_params'] = {'n_timesteps': 5, 'batch_size': 4, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True}
        po['weights_file'] = base_run_dir + 'G_model_best_weights.hdf5'
        po['D_model_params'] = {'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False, 'use_fc_precat': True, 'fc_precat_size': 1024}
        #po['model_params'] = {'n_timesteps': 5, 'batch_size': 4, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True}
        #po['weights_file'] = base_run_dir+'G_model_weights_batch16500.hdf5'
    else:
        po['model_name'] = 'facegen_rotation_prednet_twolayer'
        po['model_params'] = {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}
        po['weights_file'] = base_run_dir+'model_best_weights.hdf5'
    po['feature_layer_names'] = ['RNN']
    po['timesteps_to_use'] = [None]
    po['feature_is_time_expanded'] = [False]
    po['batch_size'] = po['model_params']['batch_size']
    po['data_file'] = '/home/bill/Data/FaceGen_Rotations/clipset5/clipsall.hkl'
    po['calculate_idx'] = range(6000)
    po['input_nt'] = 5
    po['run_num'] = -1
    po['save_dir'] = base_run_dir+'feature_analysis/perturb_features/'
    if not os.path.exists(po['save_dir']):
        os.makedirs(po['save_dir'])

    run_extract_features(po)



def fit_features(run_num):

    if run_num==307:
         base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_307/feature_analysis/'
    elif run_num==635:
         base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_635/feature_analysis/'
    elif run_num==65:
        base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/feature_analysis/'
    elif run_num==67:
        base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_67/feature_analysis_epoch0/'

    po = {}
    po['layers_to_use'] = ['RNN']
    po['layer_outputs_to_use'] = [[0]] #[[0], [0], range(5), [0], [0]]
    po['timesteps_to_use'] = [[4]]  #[[0], [0], range(5), [-1], [-1]]
    po['to_decode'] = ['pca_1', 'pca_2','pca_3','pca_4', 'pca_5', 'pan_angles', 'pan_angular_speeds', 'pan_initial_angles']
    po['fit_model'] = 'ridge'
    po['model_params'] = {'method': 'adaptive', 'start_list': [1e3,1e2,1,.1,1e-2], 'max_param': 1e5, 'min_param': 1e-5}  #ordered from most regularization to least
    po['ntrain'] = 4000
    po['nval'] = 1000
    po['ntest'] = 1000
    po['params_selection_method'] = 'best'
    po['params_file'] = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
    po['run_num'] = -1
    po['save_dir'] = base_run_dir+'perturb_decoding/'

    if not os.path.exists(po['save_dir']):
        os.mkdir(po['save_dir'])

    f = open(base_run_dir+'perturb_features/params.pkl','r')
    P_feature = pkl.load(f)
    f.close()

    run_fit_features(param_overrides=po, P_feature=P_feature)


def load_deconv_model(run_num):

    model = Graph()

    if run_num==307 or run_num==635:
        is_GAN = True
    else:
        is_GAN = False

    model.add_input(name='RNN', ndim=2)

    if is_GAN:
        model.add_input(name='random_input', ndim=2)
        model.add_node(Dense(1024+128, 64*14*14), name='fc_decoder', inputs=['RNN', 'random_input'], merge_mode='concat', concat_axis=1)
    else:
        model.add_node(Dense(1024, 64*14*14), name='fc_decoder', input='RNN')
    model.add_node(Activation('relu'), name='fc_decoder_relu', input='fc_decoder')
    model.add_node(Reshape(64, 14, 14), name='decoder_reshape', input='fc_decoder_relu')

    model.add_node(UpSample2D((3,3)), name='unpool1', input='decoder_reshape')  # (.., 32, 68, 68)
    model.add_node(Convolution2D(64, 64, 7, 7, border_mode='full'), name='deconv1', input='unpool1')  # (.., 32, 72, 72)
    model.add_node(Activation('relu'), name='deconv1_relu', input='deconv1')

    model.add_node(UpSample2D((3,3)), name='unpool2', input='deconv1_relu')  # (.., 32, 144, 144)
    model.add_node(Convolution2D(1, 64, 7, 7, border_mode='full'), name='deconv2', input='unpool2')  # (.., 32, 150, 150)
    model.add_node(Activation('relu'), name='deconv2_relu', input='deconv2')

    model.add_node(Activation('satlu'), name='predicted_frame', input='deconv2_relu') # output:  (batch_size, 1, 32, 32)

    model.add_output(name='output', input='predicted_frame')

    if run_num==307:
        params = ( ('facegen_rotation_prednet_twolayer_G_to_D', 'facegen_rotation_prednet_twolayer_D', True), ({'n_timesteps': 5, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True},{'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False}),['fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_307/G_model_weights_batch16500.hdf5' )
    elif run_num==635:
        params = ( ('facegen_rotation_prednet_twolayer_G_to_D', 'facegen_rotation_prednet_twolayer_D', True), ({'n_timesteps': 5, 'batch_size': 4, 'rand_size': 128, 'use_pixel_output': True, 'pixel_output_flattened': False, 'use_rand_input': True},{'n_timesteps': 5, 'share_encoder': True, 'encoder_params_fixed': False, 'share_RNN': False, 'RNN_params_fixed': False, 'use_fc_precat': True, 'fc_precat_size': 1024}),['fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_635/G_model_best_weights.hdf5' )
    elif run_num==65:
        params = ('facegen_rotation_prednet_twolayer', {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}, ['fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/model_best_weights.hdf5', True)
    elif run_num==67:
        params = ('facegen_rotation_prednet_twolayer', {'n_timesteps': 5, 'batch_size': 6, 'num_filt': 64, 'use_encoder_drop0': False, 'use_encoder_drop1': False, 'use_dense_drop': False}, ['fc_decoder','deconv1','deconv2'], '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_67/model_weights_epoch0.hdf5', True)

    model = initialize_weights(params, model, is_GAN=is_GAN)

    return model


def perturb_features():

    n_plot = 10
    n_perturb = 3
    time_step = 4
    is_GAN = True
    latent_vars = ['pca_1']
    #base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/'
    base_run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_635/'
    out_dir = base_run_dir+'feature_analysis/perturbation_analysis/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += 'clipset5/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    clips = hkl.load(open('/home/bill/Data/FaceGen_Rotations/clipset5/clipsall.hkl','r'))
    clips = clips[5000:]

    latent_vars_std = {}
    f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
    params_dict = pkl.load(open(f_name, 'r'))
    for v in latent_vars:
        y = get_param_values(v, params_dict, t=time_step)
        # if v=='pan_angles':
        #     k = 0.03
        # elif v=='pca_2':
        #     k = 0.06
        # else:
        #     k = 0.15
        latent_vars_std[v] = np.std(y)

    for key in params_dict:
        params_dict[key] = params_dict[key][5000:]

    f_name = base_run_dir+'feature_analysis/perturb_features/RNN_features.hkl'
    features = hkl.load(open(f_name, 'r'))
    features = features[0][5000:]

    np.random.seed(0)
    idx = np.random.permutation(features.shape[0])[:n_plot]
    features = features[idx]
    features = features[:,time_step]
    clips = clips[idx]
    for key in params_dict:
        params_dict[key] = params_dict[key][idx]

    f_name = base_run_dir+'feature_analysis/perturb_decoding/model_info.pkl'
    model_info = pkl.load(open(f_name, 'r'))

    #model = load_deconv_model(65)
    model = load_deconv_model(635)
    print 'Compiling'
    model = append_predict(model)
    #model.compile(optimizer='sgd', loss={'output': 'mse'})

    latent_vars_dx = {}
    for v in latent_vars:
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        latent_vars_dx[v] = 6*latent_vars_std[v]/np.sum(betas*betas)

    plt.figure()
    for v in latent_vars:
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        for i in range(n_plot):
            X = np.zeros((2*n_perturb+1, features.shape[1]))
            for k,j in enumerate(range(-1*n_perturb, n_perturb+1)):
                X[k] = features[i]+j*latent_vars_dx[v]*betas
                X[k][X[k]<-1] = -1
                X[k][X[k]>1] = 1


            if is_GAN:
                data = {'RNN': X, 'random_input':  np.random.uniform(low=0.0, high=0.5, size=(X.shape[0], 128))}
            else:
                data = {'RNN': X}

            predictions = model.predict(data)['output']
            if i==0:
                all_predictions = predictions
            else:
                all_predictions = np.vstack((all_predictions, predictions))

            # for j in range(3, 6):
            #     ax = plt.subplot(2, predictions.shape[0], j-2)
            #     plt.imshow(clips[i,j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            #     if j==5:
            #         #ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            #         #ax.tick_params(axis='z',which='both',bottom='off',top='off',labelbottom='off')
            #         plt.axis('off')
            #         for b in ['bottom','top','left','right']:
            #             ax.spines[b].set_color('red')
            #
            #     else:
            #         plt.axis('off')


            for j in range(predictions.shape[0]):
                plt.subplot(1, predictions.shape[0], j+1)#predictions.shape[0]
                plt.imshow(predictions[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
                plt.axis('off')
                if j==n_perturb:
                    plt.title('Perturbing '+pretty_names[v])

            plt.savefig(out_dir+'perturbplot_t'+str(time_step)+'_clip'+str(5000+idx[i])+'_'+v+'.tif')
        spio.savemat(out_dir+'perturbed_features.mat',{'predictions': all_predictions})

    plt.close('all')


def perturb_features2():

    latent_vars = ['pca_3']
    perturb_vals = np.linspace(-2, 2, 6)#[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    mult = 12

    time_step = 4
    clipset_dir = '/home/bill/Data/FaceGen_Rotations/clipset5/'
    out_dir = base_run_dir+'feature_analysis/perturbation_analysis/facegen_comparisons/mult_'+str(mult)+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    morph_dir = clipset_dir+'morphs/'
    if not os.path.exists(morph_dir):
        os.mkdir(morph_dir)

    f = open(clipset_dir+'render.xml', 'r')
    lines = f.readlines()
    f.close()

    clips = hkl.load(open('/home/bill/Data/FaceGen_Rotations/clipset5/clipsall.hkl','r'))
    clips = clips[5000:]

    latent_vars_std = {}
    f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
    params_dict = pkl.load(open(f_name, 'r'))
    for v in latent_vars:
        y = get_param_values(v, params_dict, t=time_step)
        latent_vars_std[v] = np.std(y)

    for key in params_dict:
        params_dict[key] = params_dict[key][5000:]

    f_name = base_run_dir+'feature_analysis/perturb_features/RNN_features.hkl'
    features = hkl.load(open(f_name, 'r'))
    features = features[0][5000:]

    idx = [784, 859, 906, 298, 672]
    features = features[idx]
    features = features[:,time_step]
    clips = clips[idx]
    for key in params_dict:
        params_dict[key] = params_dict[key][idx]


    f_name = base_run_dir+'feature_analysis/perturb_decoding/model_info.pkl'
    model_info = pkl.load(open(f_name, 'r'))

    model = load_deconv_model()
    print 'Compiling'
    model = append_predict(model)
    #model.compile(optimizer='sgd', loss={'output': 'mse'})

    latent_vars_dx = {}
    for v in latent_vars:
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        latent_vars_dx[v] = mult*latent_vars_std[v]/np.sum(betas*betas)

    plt.figure(figsize=(100,20))
    for vi,v in enumerate(latent_vars):
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        pca_num = int(v[-1])
        for i in range(len(idx)):

            orig_fg_file = '/home/bill/Data/FaceGen_Rotations/clipset5/fg_files/face_'+str(idx[i]+5000)+'.fg'
            ims = []
            new_vals = []
            for pi,p in enumerate(perturb_vals):
                out_fg_file = morph_dir+'face_'+str(idx[i]+5000)+'_var'+str(vi)+'_p'+str(pi)
                newp = params_dict['pca_basis'][i,pca_num-1]+p
                new_vals.append(newp)
                morph_face(orig_fg_file, out_fg_file, pca_num-1, newp)
                out_construct = morph_dir+'face_'+str(idx[i]+5000)+'_'+str(pi)
                im_file = morph_dir+'pc'+str(pca_num)+'_face_'+str(idx[i]+5000)+'_'+str(pi)+'.png'
                facegen_construct(out_fg_file, out_construct)
                lines[8] = '\t\t\t<triFilename>'+out_construct+'.tri</triFilename>\n'
                lines[9] = '\t\t\t<imgFilename>'+out_construct+'.bmp</imgFilename>\n'
                lines[28] = '\t\t<panRadians>'+str(params_dict['pan_angles'][i,time_step])+'</panRadians>\n'
                lines[82] = '\t<outputFile>'+im_file+'</outputFile>\n'
                f = open(clipset_dir+'tmp_render.xml', 'w')
                f.writelines(lines)
                f.close()
                facegen_render(clipset_dir+'tmp_render')

                src = cv.LoadImageM(im_file)
            	gray_full = cv.CreateImage(cv.GetSize(src), 8, 1)
            	grayim = cv.CreateImage((150,150), 8, 1)
            	cv.CvtColor(src, gray_full, cv.CV_BGR2GRAY)
                cv.Resize(gray_full, grayim, interpolation=cv.CV_INTER_CUBIC)
            	gray = cv.GetMat(grayim)
                ims.append(np.asarray(gray).astype('f'))


            X = np.zeros((len(perturb_vals), features.shape[1]))
            for k,j in enumerate(perturb_vals):
                X[k] = features[i]+j*latent_vars_dx[v]*betas
                X[k][X[k]<-1] = -1
                X[k][X[k]>1] = 1

            if is_GAN:
                data = {'RNN': X, 'random_input':  np.random.uniform(low=0.0, high=0.1, size=(X.shape[0], 128))}
            else:
                data = {'RNN': X}

            predictions = model.predict(data)['output']

            for j in range(predictions.shape[0]):
                # plt.subplot(1,2,1)
                # plt.imshow(ims[j]/255, cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
                # plt.axis('off')
                # plt.subplot(1,2,2)
                # plt.imshow(predictions[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
                # plt.axis('off')
                # plt.title('Perturbing '+pretty_names[v]+'\nval='+str(new_vals[j]))
                # plt.savefig(out_dir+'perturb_im'+str(idx[i]+5000)+'_'+v+'_'+str(j)+'.png')

                plt.subplot(2,len(perturb_vals),j+1)
                plt.imshow(ims[j]/255, cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
                plt.axis('off')
                plt.subplot(2,len(perturb_vals),j+1+len(perturb_vals))
                plt.imshow(predictions[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
                plt.axis('off')
                plt.title('Perturbing '+pretty_names[v]+'\nval='+str(new_vals[j]))
            plt.savefig(out_dir+v+'_perturb_im'+str(idx[i]+5000)+'.png')





def compare_perturbations():

    n_plot = 10
    time_step = 4
    out_dir = base_run_dir+'feature_analysis/perturbation_analysis/'
    latent_var = 'pan_angles'

    clips = hkl.load(open('/home/bill/Data/FaceGen_Rotations/clipset5/clipsall.hkl','r'))
    clips = clips[5000:]

    f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
    params_dict = pkl.load(open(f_name, 'r'))

    for key in params_dict:
        params_dict[key] = params_dict[key][5000:]

    f_name = base_run_dir+'feature_analysis/perturb_features/RNN_features.hkl'
    features = hkl.load(open(f_name, 'r'))
    features = features[0][5000:]

    np.random.seed(0)
    idx = np.random.permutation(features.shape[0])[:n_plot]
    features = features[idx]
    clips = clips[idx]
    for key in params_dict:
        params_dict[key] = params_dict[key][idx]

    f_name = base_run_dir+'feature_analysis/perturb_decoding/model_info.pkl'
    model_info = pkl.load(open(f_name, 'r'))

    model = load_deconv_model()
    print 'Compiling'
    model.compile(optimizer='sgd', loss={'output': 'mse'})

    tup = ('RNN', 0, time_step, latent_var)
    betas, intercept = model_info[tup]
    for i in range(n_plot):
        feat_delta = features[i,time_step]-features[i,time_step-1]
        delta_angle = params_dict[latent_var][i,time_step]-params_dict[latent_var][i,time_step-1]
        dx = delta_angle/np.sum(betas*betas)
        X_actual = features[i,:(time_step+1)]
        X_local = np.zeros((time_step+1, 1024)).astype(np.float32)
        X_linear = np.zeros((time_step+1, 1024)).astype(np.float32)
        for t in range(time_step+1):
            X_local[t] = X_actual[time_step]-(time_step-t)*feat_delta
            X_linear[t] = X_actual[time_step]+(time_step-t)*dx*betas
        angle = np.arccos(np.sum(feat_delta*betas)/(np.linalg.norm(feat_delta)*np.linalg.norm(betas)))
        print 'for clip '+str(5000+idx[i])+', angle between betas and feat_delta: '+str(angle)
        # for _ in range(10):
        #     vec = np.random.rand(1024)
        #     angle = np.arccos(np.sum(feat_delta*vec)/(np.linalg.norm(feat_delta)*np.linalg.norm(vec)))
        #     print 'for clip '+str(5000+idx[i])+', angle between betas and random vec: '+str(angle)

        predictions_local = model.predict({'RNN': X_local})['output']
        predictions_actual = model.predict({'RNN': X_actual})['output']
        predictions_linear = model.predict({'RNN': X_linear})['output']
        vals_real = params_dict[latent_var][i,:time_step+1]
        vals_local = np.zeros(time_step+1)
        vals_actual = np.zeros(time_step+1)
        vals_linear = np.zeros(time_step+1)
        for t in range(time_step+1):
            vals_local[t] = np.sum(X_local[t]*betas)
            vals_actual[t] = np.sum(X_actual[t]*betas)
            vals_linear[t] = np.sum(X_linear[t]*betas)


        for j in range(time_step+1):
            plt.subplot(4, time_step+1, j+1)
            plt.imshow(clips[i,j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.axis('off')
            if j==0:
                plt.title('actual frames')
            else:
                plt.title(np.round(100*vals_real[j])/100)

            plt.subplot(4, time_step+1, j+2+time_step)
            plt.imshow(predictions_actual[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.axis('off')
            if j==0:
                plt.title('actual features')
            else:
                plt.title(np.round(100*vals_actual[j])/100)

            plt.subplot(4, time_step+1, j+3+time_step*2)
            plt.imshow(predictions_local[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.axis('off')
            if j==0:
                plt.title('local linear projection')
            else:
                plt.title(np.round(100*vals_local[j])/100)

            plt.subplot(4, time_step+1, j+4+time_step*3)
            plt.imshow(predictions_linear[j,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.axis('off')
            if j==0:
                plt.title('global linear projection')
            else:
                plt.title(np.round(100*vals_linear[j])/100)

        plt.savefig(out_dir+'comparison_perturbplot_t'+str(time_step)+'_clip'+str(5000+idx[i])+'_'+latent_var+'.tif')


def make_param_scale_plots():

    out_dir = '/home/bill/Dropbox/Cox_Lab/Predictive_Networks/misc/facegen_param_plots/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    latent_vars = ['pca_1', 'pca_2', 'pca_3', 'pan_angles']

    clips = hkl.load(open('/home/bill/Data/FaceGen_Rotations/clipset5/clipsall.hkl','r'))

    f_name = '/home/bill/Data/FaceGen_Rotations/clipset5/all_params_all.pkl'
    params_dict = pkl.load(open(f_name, 'r'))

    k = 0.4
    n_perturb = 5
    for v in latent_vars:
        y = get_param_values(v, params_dict, t=4)
        mu = y.mean()
        sigma = y.std()

        #idx = []
        for j,i in enumerate(range(-n_perturb, n_perturb+1)):
            target = mu+k*i*sigma
            vals = np.abs(y-target)
            #idx.append(np.argmin(vals))
            idx = np.argmin(vals)
            plt.subplot(1,n_perturb*2+1, j+1)
            plt.imshow(clips[idx,4,0], cmap="Greys_r", vmin=0.0, vmax=1.0, interpolation='none')
            plt.axis('off')

        plt.savefig(out_dir+'variations_in_'+v+'.jpg')




def create_pertubation_dataset():

    run_num = 635
    latent_vars = ['pca_3']
    perturb_vals = np.linspace(-2, 2, 5)
    mult = 12
    time_step = 4
    faces_to_use = range(20)
    angles_to_use = range(0,13,2)

    clipset = 10
    version_map = {10: 2}
    clipset_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/'
    if run_num==307:
        run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_307/'
        is_GAN = True
    elif run_num==635:
        run_dir = '/home/bill/Projects/Predictive_Networks/facegen_GAN_runs_server/run_635/'
        is_GAN = True
    elif run_num==65:
        run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_65/'
        is_GAN = False
    elif run_num==67:
        run_dir = '/home/bill/Projects/Predictive_Networks/facegen_runs_server/run_67/'
        is_GAN = False

    if run_num==67:
        f_str = '_epoch0'
    else:
        f_str = ''

    feature_dir = run_dir +'/feature_analysis'+f_str+'/classification/version_'+str(version_map[clipset])+'/features/'

    clips = hkl.load(open(clipset_dir+'clipsall.hkl','r'))
    params_dict = pkl.load(open(clipset_dir+'face_params.pkl','r'))
    features = hkl.load(open(feature_dir+'RNN_features.hkl','r'))[0][:,time_step]
    if run_num==67:
        f_name = run_dir+'feature_analysis'+f_str+'/decoding/model_info.pkl'
    else:
        f_name = run_dir+'feature_analysis'+f_str+'/perturb_decoding/model_info.pkl'
    model_info = pkl.load(open(f_name, 'r'))
    clipset_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/'
    f = open(clipset_dir+'params.pkl','r')
    d = pkl.load(f)
    f.close()

    face_labels = d['face_labels']
    pan_angle_labels = d['pan_angle_labels']
    idx = [i for i in range(len(face_labels)) if (face_labels[i] in faces_to_use and pan_angle_labels[i] in angles_to_use)]
    features = features[idx]
    face_labels = face_labels[idx]
    angle_labels = pan_angle_labels[idx]

    model = load_deconv_model(run_num)
    print 'Compiling'
    model = append_predict(model)

    latent_vars_dx = {}
    for v in latent_vars:
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        latent_vars_dx[v] = mult*1.0/np.sum(betas*betas)

    predictions = {}
    for vi,v in enumerate(latent_vars):
        tup = ('RNN', 0, time_step, v)
        betas, intercept = model_info[tup]
        pca_num = int(v[-1])

        X = np.zeros((len(perturb_vals)*features.shape[0], features.shape[1]))
        count = 0
        for i in range(features.shape[0]):
            for k,j in enumerate(perturb_vals):
                vals = features[i]+j*latent_vars_dx[v]*betas
                vals[vals<-1] = -1
                vals[vals>1] = 1
                X[count] = vals
                count += 1

        if is_GAN:
            data = {'RNN': X, 'random_input':  np.random.uniform(low=0.0, high=0.5, size=(X.shape[0], 128))}
        else:
            data = {'RNN': X}
        predictions[v] = model.predict(data,batch_size=10,verbose=1)['output']
        #predictions[v] = predictions[v].reshape((features.shape[0],))

    out_dir = run_dir+'feature_analysis'+f_str+'/perturbation_analysis/clipset'+str(clipset)+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir += 'mult_'+str(mult)+'/'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    spio.savemat(out_dir+'perturbed_features.mat',predictions)
    spio.savemat(out_dir+'perturbed_params.mat',{'perturb_vals': perturb_vals, 'mult': mult, 'faces_to_use': faces_to_use, 'angles_to_use': angles_to_use, 'face_labels': face_labels, 'angle_labels': angle_labels})


def create_facegen_morphs():

    latent_vars = ['pca_1', 'pca_2','pca_3']
    perturb_vals = np.linspace(-2, 2, 5)
    faces_to_use = range(20)

    clipset = 10

    clipset_dir = '/home/bill/Data/FaceGen_Rotations/clipset'+str(clipset)+'/'
    f = open(clipset_dir+'params.pkl','r')
    d = pkl.load(f)
    P = d['P']
    f.close()
    pan_angles = P['pan_angles']
    params_dict = pkl.load(open(clipset_dir+'face_params.pkl','r'))
    f = open(clipset_dir+'render.xml', 'r')
    lines = f.readlines()
    f.close()

    pan_angles = pan_angles[::2]

    morph_dir = clipset_dir+'morphs/'
    if not os.path.exists(morph_dir):
        os.makedirs(morph_dir)
    out_im_dir = morph_dir + 'images/'
    if not os.path.exists(out_im_dir):
        os.makedirs(out_im_dir)

    for i in faces_to_use:
        orig_fg_file = clipset_dir+'fg_files/face_'+str(i)+'.fg'
        for vi,v in enumerate(latent_vars):
            pca_num = int(v[-1])
            for pi,p in enumerate(perturb_vals):
                out_fg_file = morph_dir+'face_'+str(i)+'_pc'+str(pca_num)+'_p'+str(pi)
                newp = params_dict['pca_basis'][i,pca_num-1]+p
                morph_face(orig_fg_file, out_fg_file, vi, newp)
                out_construct = morph_dir+'face_'+str(i)+'_pc'+str(pca_num)+'_p'+str(pi)+'_construct'
                facegen_construct(out_fg_file, out_construct)
                lines[8] = '\t\t\t<triFilename>'+out_construct+'.tri</triFilename>\n'
                lines[9] = '\t\t\t<imgFilename>'+out_construct+'.bmp</imgFilename>\n'
                for ai, angle in enumerate(pan_angles):
                    im_file = out_im_dir+'face_'+str(i)+'_pc'+str(pca_num)+'_p'+str(pi)+'_angle'+str(ai)+'.png'
                    lines[28] = '\t\t<panRadians>'+str(angle)+'</panRadians>\n'
                    lines[82] = '\t<outputFile>'+im_file+'</outputFile>\n'
                    f = open(clipset_dir+'tmp_render.xml', 'w')
                    f.writelines(lines)
                    f.close()
                    facegen_render(clipset_dir+'tmp_render')


if __name__=='__main__':
    try:
        #create_features(True)
        #fit_features(635)
        #perturb_features()
        #compare_perturbations()
        #make_param_scale_plots()
        create_pertubation_dataset()
        #create_facegen_morphs()


    except:
        ty, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
