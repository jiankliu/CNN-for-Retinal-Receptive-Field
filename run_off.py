"""
Deepretina example script
"""

from __future__ import absolute_import
from keras.models import Model, model_from_json

from core.models_off import sequential,nips_conv
from core.load_data_ori import DataSet
from core.train import train
from core.monitor import Monitor
import numpy as np
from time import time
import tableprint as tp
import os,h5py
from core.visualization import DropBox

def train_model(rootpath, h5path, data_type, stim_shape = None, cell_name = None, num_filters=32, cfg=None):
    print('*********** Training *************')
    batchsize = 2000
    num_epochs = 3000
    num_iters = 50000
    #num_epochs = 200

    layers = retina_conv(num_cells=1, stim_shape=stim_shape, batch_size = batchsize, num_filters=num_filters)

    # compile the keras model
    model = sequential(layers, 'adam', loss='poisson')

    # load experiment data
    data = DataSet(rootpath, h5path, stim_shape[0], batchsize, cfg=cfg)
    monitor = Monitor(data_type, rootpath, model, data, cell_name)

    # train
    max_iters = train(model, data, monitor, num_iters, num_epochs)

    return max_iters

def test_model(rootpath, h5path, data_type, stim_shape, cell_name, rd_idx = None, iters=None, cfg=None):
    print('*********** Testing *************')

    nt = stim_shape[0]
    nh = stim_shape[1]
    nw = stim_shape[2]

    to_cell = ''
    db = DropBox(rootpath, h5path, data_type, cell_name, nt=nt, iters=iters, toCell=to_cell, cfg=cfg)

    cc = db.draw_test_response()

    db.draw_train_STA()
    db.visualize_filter(db.model.get_weights()[0], layer_name='layer_0')

    layer_id = 0
    sta = db.get_sta(layer_id, samples=10000, batch_size=2000, stim_shape=(nt, nh, nw))
    db.visualize_sta(sta, img_name='sta_filter' + str(layer_id))

    #return cc

def cfg_set(cfg_id=-1):
    if cfg_id == -1:
        cfg = None
    else:
        cfg = {
                'noise': False,
                'noise_ratio': 0.0,
                'percent': False,
                'percent_ratio': 0.0,
                'batch': False,
                'batch_size': 2000
                }
        if cfg_id == 0:
            cfg['noise'] = True
        elif cfg_id == 1:
            cfg['percent'] = True
        elif cfg_id == 2:
            cfg['batch'] = True

    return cfg

if __name__ == '__main__':

    i = 1 #time
    #nt = 20 if i == 0 else 1
    nt = 20
    rootpath = os.path.abspath(os.curdir)
    datafold = rootpath + '/data/'

    num_filters_list = [1]

    cfg = cfg_set(1)
    per_arr = [0.005]
    

    for conv_filters in num_filters_list:
    
        for per_ratio in per_arr:

            cnt = 0

            cell_id = 'off_cell' 
            h5fold = datafold + 'h5py/' + cell_id + '/'
            output_data_type =  cell_id + str(conv_filters) + '_' + str(per_ratio * 100)

            print('saving result to ' + output_data_type)

            h5path = h5fold + 'whitenoise-' + cell_id +'/'
            with h5py.File(h5path+'config.hdf5', mode='r') as f:
                nh = np.array(f['shape'])[0]
                nw = np.array(f['shape'])[1]

            batchsize = 5000

            stim_shape = (nt, nh, nw)
            #stim_shape = (batchsize, nt, nh, nw)
            print ('stim_shape: ', str(stim_shape))
            cell_name = cell_id
            print ('*********** ', cell_name, ' ***********')
            recordpath = rootpath + '/output/'+ output_data_type +'/record/'+ cell_name +'/'
                
            cfg['percent_ratio'] = round(per_ratio, 5)
            end_iter = 16000

            for iteration in range(2000, end_iter+1, 2000):
                print('Testing the model of iteration ', str(iteration))

                test_model(rootpath, h5path, output_data_type, stim_shape=stim_shape, cell_name=cell_name, iters=iteration, cfg=cfg)

            test_model(rootpath, h5path, output_data_type, stim_shape=stim_shape, cell_name=cell_name, cfg=cfg)


