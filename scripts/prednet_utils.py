from prednet2 import append_predict, evaluate_prednet
from keras_models import initialize_GAN_models, initialize_model
import pickle as pkl
import hickle as hkl


def load_prednet_model(P, weights_file):

    if isinstance(P, str):
        P = pkl.load(open(P, 'r'))

    if P['is_GAN']:
        model = {}
        model['G'], model['D'] = initialize_GAN_models(P['G_model_name'], P['D_model_name'], P['G_model_params'], P['D_model_params'])
        if isinstance(weights_file, str):
            model['G'].load_weights(weights_file)
            model = model['G']
        else:
            for key in weights_file:
                model[key] = model[key].load_weights(weights_file[key])
    else:
        model = initialize_model(P['model_name'], P['model_params'])
        model.load_weights(weights_file)

    return model


def get_prednet_predictions(P, model, data_file, n_predict):

    if isinstance(P, str):
        P = pkl.load(open(P, 'r'))

    model = append_predict(model)

    P['evaluation_file'] = data_file
    P['n_evaluate'] = n_predict

    predictions, actual_sequences, pre_sequences = evaluate_prednet(P, model, True)

    return predictions, actual_sequences


def get_data_size(data_file):

    X = hkl.load(open(data_file,'r'))

    return X.shape
