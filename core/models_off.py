"""
Construct Keras models
"""

from __future__ import absolute_import, division, print_function

from keras.models import Sequential
from keras.layers.core import Dropout, Dense, Activation, Flatten, TimeDistributedDense
from keras.layers.core import Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.advanced_activations import PReLU, ParametricSoftplus 
#ThresholdedReLU 
#YQSoftplus
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.regularizers import l1l2, activity_l1l2, l2
from .metrics import notify

def sequential(layers, optimizer, loss='poisson'):
    """Compiles a Keras model with the given layers

    Parameters
    ----------
    layers : list
        A list of Keras layers, in order

    optimizer : string or optimizer
        Either the name of a Keras optimizer, or a Keras optimizer object

    loss : string, optional
        The name of a Keras loss function (Default: 'poisson_loss'), or a
        Keras objective object

    Returns
    -------
    model : keras.models.Sequential
        A compiled Keras model object
    """
    model = Sequential(layers)
    with notify('Compiling'):
        model.compile(loss=loss, optimizer=optimizer)
    return model

def retina_conv(num_cells, stim_shape, batch_size, num_filters):

    layers = list()
    #stim_shape = list(stim_shape); 
    #input_shape = stim_shape 
    input_shape = stim_shape
    #batch_input_shape = stim_shape

    # injected noise strength
    sigma = 0.1

    convlayers = [(num_filters, 7)]

    l1_weight = 5.5e-4
    l2_weight = 1e-3

    # weight and activity regularization
    W_reg = [(l1_weight, l2_weight), (l1_weight, l2_weight), (l1_weight, l2_weight)]
    act_reg = [(0., 0.), (0., 0.), (0., 0.)]

    l2_bias = 1e-3

    # loop over convolutional layers
    for (n, size), w_args, act_args in zip(convlayers, W_reg, act_reg):
        args = (n, size, size)
        kwargs = {
            'border_mode': 'valid',
            'subsample': (1, 1),
            'init': 'normal',
            'W_regularizer': l1l2(*w_args),
            'activity_regularizer': activity_l1l2(*act_args),
            'dim_ordering': 'th'
        }
        if len(layers) == 0:
            #print('input_shape: %s' % str(input_shape))
            #kwargs['batch_input_shape'] = batch_input_shape
            kwargs['input_shape'] = input_shape
            #kwargs['batch_size'] = batch_size


        # add convolutional layer
        layers.append(Convolution2D(*args, **kwargs))

        # add gaussian noise
        layers.append(GaussianNoise(sigma))

        # add ReLu
        layers.append(Activation('relu'))

    # flatten
    layers.append(Flatten())

    # Add a final dense (affine) layer
    layers.append(Dense(num_cells, init='normal',
                        W_regularizer=l1l2(0., l2_weight),
                        activity_regularizer=activity_l1l2(1e-3, 0.)))

    layers.append(ParametricSoftplus())

    return layers


