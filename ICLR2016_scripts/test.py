import theano, sys
sys.path.append('/home/bill/Libraries/keras/')
from keras.layers.core import Layer

class ExpandTimesteps2(Layer):
    '''
        Make (nb_samples*timestep, *dims) be (nb_samples, timesteps, *dims)
        where first n_timestep elements are from the first sample
    '''
    def __init__(self, batch_size=None, n_timesteps=None):
        super(ExpandTimesteps2, self).__init__()
        if batch_size is not None:
            self.division_param = batch_size
            self.use_batch_size = True
        else:
            self.division_param = n_timesteps
            self.use_batch_size = False

    def get_output(self, train):
        X = self.get_input(train)
        if self.use_batch_size:
            nt = X.shape[0]/self.division_param
        else:
            nt = self.division_param
        nshape = (X.shape[0]/nt, nt) + X.shape[1:]
        return theano.tensor.reshape(X, nshape)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "division_param":self.division_param,
            "use_batch_size":self.use_batch_size}

class TestClass():

    def __int__(self):
        self.var = 5

tmp = TestClass()
#tmp = ExpandTimeSteps2(batch_size=128)
