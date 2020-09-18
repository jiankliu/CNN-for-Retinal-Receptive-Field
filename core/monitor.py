"""
Helper utilities for saving models and model outputs
"""

from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')

from os import mkdir, uname, getenv, path, makedirs
from json import dumps
from collections import namedtuple
from itertools import product
from functools import wraps
from .metrics import notify, allmetrics
from warnings import warn
from keras.models import model_from_json
import numpy as np
import inspect
import subprocess
import shutil
import time
import keras
import deepretina
import hashlib
import h5py,os
# Force matplotlib to not use any X-windows with the Agg backend
import matplotlib.pyplot as plt

class Monitor:
    def __init__(self, data_type, rootpath, model, data, cell_name, cfg=None):
        """Monitor base class

        Parameters
        ----------
        name : str
            A short string describing this model

        model : object
            A Keras or GLM model object

        experiment : experiments.Experiment
            A pointer to an Experiment class used to grab test data

        save_interval : int
            Parameters are saved only every save_interval iterations
        """
        self.data_type = data_type
        self.rootpath = rootpath
        outpath = self.rootpath + '/output/'+ self.data_type +'/record/'
        self.path = outpath + cell_name + '/' #+ '--' + time.strftime("%Y-%m-%d_%H:%M:%S")+'/'
        if not cfg == None:
            if cfg['noise']:
                self.path = self.path + 'noise/' + str(cfg['noise_ratio']) + '/'
            elif cfg['percent']:
                self.path = self.path + 'percent/' + str(cfg['percent_ratio']) + '/'
            elif cfg['batch']:
                self.path = self.path + 'batch/' + str(cfg['batch_size']) + '/'

        make_folder(self.path)

        self.model = model
        self.data = data
        self.metrics = ('cc',)
                    
        filepath = path.join(self.path,'result/')
        make_folder(filepath)
        filename = path.join(filepath,'performance.h5')
        h5py.File(filename, mode='w')
        
        filepath = path.join(self.path,'model/')
        make_folder(filepath)
        filename = path.join(filepath,'architecture.json')
        model_json = model.to_json()
        open(filename,'w').write(model_json)
        
        filepath = path.join(self.path,'display/')
        make_folder(filepath)
        
    
    def save_model(self, epoch, iters, model, is_final = False):
        """Saves relevant information for this epoch/iteration of training"""
        self.model = model
        filepath = path.join(self.path,'model/')
        if (is_final):
            filename = path.join(filepath,'weights_final.h5')
            model.save_weights(filename)

        else:
            filename = path.join(filepath, 'weights_' + str(iters) + '.h5')
            model.save_weights(filename)
        
        
    def save_performance(self, iters, loss, avg_val, all_val):
        #save result
        filepath = path.join(self.path,'result/')
        filename = path.join(filepath,'performance.h5')
        with h5py.File(filename, mode='r+') as f:
            f.create_dataset('iters',data = iters)
            f.create_dataset('loss',data = loss)
            
            g_avg = f.create_group('avg_val')            
            dict_avg = {}
            for item in avg_val:              
                for key, value in item.items():
                    dict_avg[key] = []
                break
            for item in avg_val:              
                for key, value in item.items():
                    dict_avg[key].append(value)          
            for key, value in dict_avg.items():
                g_avg.create_dataset(key,data = value)
                
            g_all = f.create_group('all_val')           
            dict_all = {}
            for item in all_val:
                for key, value in item.items():
                    dict_all[key] = []
                break
            for item in all_val:
                for key, value in item.items():
                    dict_all[key].append(value)                   
            for key, value in dict_all.items():
                g_all.create_dataset(key,data = value)

        
        typlist = ['loss','avg_val']
        for curve_type in typlist :
            filepath = path.join(self.path,'display/')
            filename = path.join(filepath, curve_type)
            self._plot_curve(curve_type, filename)

    def _plot_curve(self, curve_type, filename):
        
        # plot the performance curves
        datapath = path.join(self.path, 'result/performance.h5')
        if curve_type == 'loss': 
            with h5py.File(datapath, mode='r') as f:
                x = np.array(f[curve_type])
                it = np.array(f['iters'])
                it.astype(int)

            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            ax.plot(it, x)
            #ax.set_title('Loss')
            ax.set_xlabel('Iters', fontsize=16)
            ax.set_ylabel(curve_type, fontsize=16)
            fig.savefig(filename)
            
        elif curve_type == 'avg_val':
            fig, axs = plt.subplots(4, 1, figsize=(16, 10))
            with h5py.File(datapath, mode='r') as f:
                it = np.array(f['iters'])
                it.astype(int)
                for i in range(len(self.metrics)):
                    ax = axs[i]
                    func = self.metrics[i]
                    x = np.array(f[curve_type][func])
                    ax.plot(it, x)
                    ax.set_xlabel('Iters', fontsize=16)
                    ax.set_ylabel(func, fontsize=16)
            fig.savefig(filename)
            
        elif curve_type == 'all_val':           
            fig, axs = plt.subplots(4, 1, figsize=(16, 10))
            with h5py.File(datapath, mode='r') as f:
                it = np.array(f['iters'])
                it.astype(int)
                for i in range(len(self.metrics)):
                    ax = axs[i]
                    func = self.metrics[i]
                    x = np.array(f[curve_type][func])
                    num = len(x[0])
                    for no in range(num):
                        ax.plot(it, x[:,no],label = '$'+'Neuron '+str(no)+'$')
                    ax.set_xlabel('Iters', fontsize=16)
                    ax.set_ylabel(func, fontsize=16)
            fig.savefig(filename)
            
        else:
             print("\nThe type of the curve is incorrect")

    
    def getpath(self, filename):
        """Generates a full path to save the given file in the database directory"""
        return path.join(self.path, filename)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        import shutil
        shutil.rmtree(path)
