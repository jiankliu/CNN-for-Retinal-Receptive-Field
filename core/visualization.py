"""Visualize the filters"""

from __future__ import absolute_import, division, print_function
import numpy as np
import pyret.filtertools as ft #Tools and utilities for computing spike-triggered averages (filters),
import theano
import os
import h5py, random

import matplotlib
matplotlib.use('Agg')

from .metrics import allmetrics

from keras.models import model_from_json

#%matplotlib inline
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from pylab import rcParams
rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'gray'
iters_w = -1
iters_f = -1

class DropBox:
    def __init__(self, rootpath, h5path, data_type, cell_name, nt, iters = None, toCell = '', cfg=None, 
                    prune=None, rd_idx=None, stim_type=None):
        self.path = rootpath + '/output/'+ data_type +'/visualize/'+ cell_name +'/'
        self.cfg = cfg
        '''
        if not cfg == None:
            if cfg['noise']:
                self.path = self.path + 'noise/' + str(cfg['noise_ratio']) + '/'
            elif cfg['percent']:
                self.path = self.path + 'percent/' + str(cfg['percent_ratio']) + '/'
            elif cfg['batch']:
                self.path = self.path + 'batch/' + str(cfg['batch_size']) + '/'
        '''

        self.recordpath = rootpath + '/output/'+ data_type +'/record/'+ cell_name +'/'
        '''
        if not cfg == None:
            if cfg['noise']:
                self.recordpath = self.recordpath + 'noise/' + str(cfg['noise_ratio']) + '/'
            elif cfg['percent']:
                self.recordpath = self.recordpath + 'percent/' + str(cfg['percent_ratio']) + '/'
            elif cfg['batch']:
                self.recordpath = self.recordpath + 'batch/' + str(cfg['batch_size']) + '/'
        '''

        self.rootpath = rootpath
        self.h5path = h5path

        if stim_type == None:
            self.stim_type = ''
        else:
            self.stim_type = stim_type

        if toCell == '':
            self.toCell = toCell
        else:
            self.toCell = '_'+toCell

        self.cell_name =cell_name
        if iters == None:
            self.iters = 'final'
        else:
            self.iters = iters
        self.model = self.read_model(self.iters)
        #self.model = self.read_prune_model(self.iters, rd_idx)
        self.nt = nt
        #self.visualpath = self.path + prune + str(rd_idx) + 'visual_cell'+ self.toCell+'_' + str(self.iters) +'/'
        #self.visualpath = self.path + prune + 'visual_cell'+ self.toCell+'_' + str(self.iters) +'/'
        self.visualpath = self.path + 'visual_cell'+ self.toCell+'_' + str(self.iters) +'/'

        if os.path.exists(self.visualpath) == False:
            os.makedirs(self.visualpath)

    def read_prune_model(self, iters = 500, rd_idx=0):

        print('Testing prune model..')
        #filename = os.path.join(self.recordpath,'model/architecture_prune_' +str(rd_idx) + '.json')
        filename = os.path.join(self.recordpath,'model/architecture_prune_spaCorr' + '.json')

        json_string = open(filename,'r').read()
        model = model_from_json(json_string)

        #filename = os.path.join(self.recordpath,'model/prune_weights_final_'+ str(rd_idx) +'.h5')
        filename = os.path.join(self.recordpath,'model/spaCorr_prune_weights_final' +'.h5')

        #print('filename ' + filename)
        model.load_weights(filename)
        return model


    def read_model(self, iters = 500):

        print('Testing the orginal model')
        filename = os.path.join(self.recordpath,'model/architecture.json')
        json_string = open(filename,'r').read()
        model = model_from_json(json_string)

        filename = os.path.join(self.recordpath,'model/weights_'+str(iters)+'.h5')
        #print('filename ' + filename)
        model.load_weights(filename)
        return model

    def visualize_filter(self, weights, title='convnet', layer_name='layer_0', fig_size=(8,10), dpi=300, cmap='seismic'):

        #print (weights.shape)
        num_filters = weights.shape[0]

        # plot space and time profiles together
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title(title, fontsize=20)
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))

        cl = 0
        for x in range(num_rows):
            for y in range(num_cols):
                plt_idx = x * num_cols + y + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    if self.nt>1:
                        spatial,temporal = ft.decompose(weights[plt_idx-1])
                    else:
                        spatial = weights[plt_idx-1][0]
                    cl = max(np.max(abs(spatial)), cl)
        colorlimit = [-cl, cl]

        for x in range(num_rows):
            for y in range(num_cols):
                plt_idx = x * num_cols + y + 1
                # in case fewer weights than fit neatly in rows and cols
                if plt_idx <= len(weights):
                    if self.nt > 1:
                        spatial,temporal = ft.decompose(weights[plt_idx-1])
                    else:
                        spatial = weights[plt_idx-1][0]
                        temporal = np.array([])
                    ax = plt.subplot2grid((num_rows*4, num_cols), (4*x, y), rowspan=3)
                    ax.imshow(spatial, interpolation='nearest', cmap=cmap, clim=colorlimit)
                    #plt.title('Subunit %i' %plt_idx)
                    plt.grid('off')
                    plt.axis('on')
                    plt.xticks([])
                    plt.yticks([])

                    ax = plt.subplot2grid((num_rows*4, num_cols), (4*x+3, y), rowspan=1)
                    ax.plot(np.linspace(0,len(temporal)*10,len(temporal)), temporal, 'k', linewidth=2)
                    plt.grid('off')
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
        
        plt.suptitle('Iteration ' + str(self.iters))
        plt.savefig(self.visualpath + layer_name + '_filter' + str(self.iters) +'.png', dpi=dpi)
        #plt.savefig(self.visualpath + layer_name + '_filter.eps', dpi=dpi)
        plt.savefig(self.visualpath + layer_name + '_filter.eps')

        plt.close()

    def animation_filter(self, num_iters = 5, layer_name='layer_0', fig_size=(8,10)):

        print ('****************Animation Weights****************')
        #get weight shape
        weights = self.model.get_weights()[0]
        num_filters = weights.shape[0]

        interval = 500;
        num_cols = int(np.sqrt(num_filters))
        num_rows = int(np.ceil(num_filters/num_cols))

        weights_list  = []
        for i in range(num_iters-1):
            model = self.read_model(iters= i*interval)
            weights_list.append(model.get_weights()[0])

        weights_max = np.max(abs(np.array(weights_list)))
        colorlimit = [-weights_max, weights_max]

        # plot gif

        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        def make_frame(t):
            global iters_w
            if (iters_w >=0 and iters_w < num_iters-1):
                print ('Animate weights',iters_w)
                for x in range(num_rows):
                    for y in range(num_cols):
                        plt_idx = x * num_cols + y + 1
                        if plt_idx <= num_filters:
                            spatial = weights_list[iters_w][plt_idx - 1][0]
                            ax = plt.subplot2grid((num_rows*4, num_cols), (4*x, y), rowspan=3)
                            ax.imshow(spatial, interpolation='nearest', cmap='seismic', clim=colorlimit)
                            ax.set_title('Filter %i' %plt_idx)
                            plt.grid('off')
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('on')
                        #if plt_idx <= num_filters:
                space = np.zeros_like(spatial)
                ax = plt.subplot2grid((num_rows*4, num_cols), (4*x, y), rowspan=3)
                ax.imshow(space, interpolation='nearest', cmap='seismic', clim=colorlimit)
                plt.grid('off')
                plt.axis('off')
                iters_show = iters_w*interval
                plt.text(-40, 10, 'Iteration %i'%iters_show, fontdict={'size': 24, 'color': 'black'})


            iters_w += 1
            return mplfig_to_npimage(fig)

        fps = 5
        duration = num_iters/fps
        animation = VideoClip(make_frame, duration=duration)
        animation.write_gif(self.visualpath + layer_name + '_filter_film.gif', fps=fps)

    def draw_test_response(self):
        filepath = os.path.join(self.h5path, self.stim_type+'test'+self.toCell+'.hdf5')
        nt = self.nt
        with h5py.File(filepath, 'r') as f:
            X = np.array(f['X'])
            if nt > 1:
                X = rolling_window(X, nt, time_axis=0)
                y = np.array(f['y'])[nt:]
                r = np.array(f['r'])[nt:]
            else:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                y = np.array(f['y'])
                r = np.array(f['r'])
        #print(y.shape)

        metrics = ('cc',)

        extract_len = len(r)

        X_ex = X[:extract_len]

        num_trials = y.shape[1]
        #print('num_trails' + str(num_trials)) 
        #print('y shape: ' + str(y.shape))
        #test_trials = y.shape[1]
        #ind_test_trials = np.array(range(y.shape[1]))
        #random.shuffle(ind_test_trials)
        y_show = y
        for i in range(y.shape[1]):
            y_show[:,i] = np.array((y[:, i]) > 0)

        r_pre = self.model.predict_on_batch(X_ex)[0]
        y_pre = np.zeros((num_trials,)+r_pre.shape)

        for i in range(num_trials):
            y_pre[i] = np.random.poisson(r_pre)
        y_pre_show = np.array(y_pre[:,:extract_len,0]>0)
        r_show = r[:extract_len,0]
        r_pre_show = r_pre[:extract_len,0]

        avg_val, all_val = allmetrics(r_show, r_pre_show, metrics)
        avg = avg_val['cc']
        write_txt(self.path +'cc.txt', 'model-'+self.cell_name+'-test-'+self.toCell+'  '+str(avg))

        plt.subplot(3,1,1)
        y_show = y_show.T
        plt.title('Cell responses to white noise\nPearson CC = '+str(avg)+'\n', fontsize=20)

        time = np.array(range(extract_len))
        #plt.imshow(y_show.T, aspect='auto', cmap=cm.gray_r)
        s = 4
        line = 3
        timeshow = extract_len
        for i in range(num_trials):
            spike = np.where(y_show[i, :]>0)
            plt.scatter(time[spike], y_show[i, spike]*i, s=s, marker='|', color='b')
        plt.xlim([0,timeshow])
        plt.ylabel('Data', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(),[])#, ['left','bottom'])

        plt.subplot(3,1,2)
        for i in range(num_trials):
            spike = np.where(y_pre_show[i, :]>0)
            plt.scatter(time[spike], y_pre_show[i, spike]*i, s=s, marker='|', color='r')

        plt.xlim([0,timeshow])
        plt.ylabel('CNN', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), [])

        r_show_max = np.max(r_show)
        r_show = r_show / r_show_max
        #print('r_show_max' + str(r_show_max))

        r_pre_show_max = max(r_pre_show)
        r_pre_show = r_pre_show / r_pre_show_max


        plt.subplot(3,1,3)
        plt.plot(r_show.T * 30, 'b', linewidth=line, alpha=0.4, label='data')
        plt.plot(r_pre_show.T * 30, 'r', linewidth=line, alpha=0.4, label='CNN prediction')
        #plt.legend(loc='upper left')
        plt.ylabel('Rate', fontsize=18)
        plt.xlim([0,timeshow])

        to_xticks = [str(e*2) for e in range(6)]
        plt.xticks(np.array(range(6))*60, to_xticks)

        #plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left','bottom'])

        plt.savefig(self.visualpath+'respose_test.png')
        plt.savefig(self.visualpath+'respose_test.eps')
        #plt.savefig(self.visualpath+'respose_test.svg')
        plt.close()

        filepath = self.visualpath + 'record.h5'
        with h5py.File(filepath, 'w') as fw:
            #fw.create_dataset('X_input', data=X_ex)
            fw.create_dataset('y_data_show', data=y_show)
            fw.create_dataset('r_data_show', data=r_show)
            fw.create_dataset('y_pre_show', data=y_pre_show)
            fw.create_dataset('r_pre_show', data=r_pre_show)

        return avg

    def draw_nature_test_response(self, filename):
        filepath = os.path.join(self.h5path, filename + self.toCell+'.hdf5')
        nt = self.nt
        with h5py.File(filepath, 'r') as f:
            X = np.array(f['X'])
            if nt > 1:
                X = rolling_window(X, nt, time_axis=0)
                y = np.array(f['y'])[nt:]
                r = np.array(f['r'])[nt:]
            else:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                y = np.array(f['y'])
                r = np.array(f['r'])
#        print(y.shape)
#        print(r.shape)

        metrics = ('cc',)

        extract_len = len(r)

        X_ex = X[:extract_len]

        num_trials = y.shape[1]
        #print('num_trails' + str(num_trials)) 
        #print('y shape: ' + str(y.shape))
        #test_trials = y.shape[1]
        #ind_test_trials = np.array(range(y.shape[1]))
        #random.shuffle(ind_test_trials)
        y_show = y
        for i in range(y.shape[1]):
            y_show[:,i] = np.array((y[:, i]) > 0)

        r_pre = self.model.predict_on_batch(X_ex)[0]
        y_pre = np.zeros((num_trials,)+r_pre.shape)

        for i in range(num_trials):
            y_pre[i] = np.random.poisson(r_pre)
        y_pre_show = np.array(y_pre[:,:extract_len,0]>0)
        #r_show = r[:extract_len,0]
        r_show = r[:extract_len]
        r_pre_show = r_pre[:extract_len,0]
        #print(r_show)
        #print(r_pre_show)

        avg_val, all_val = allmetrics(r_show, r_pre_show, metrics)
        avg = avg_val['cc']
        write_txt(self.path + filename + '_cc.txt', 'model-'+self.cell_name+'-test-'+self.toCell+'  '+str(avg))

        plt.subplot(3,1,1)
        y_show = y_show.T
        #plt.title('Cell responses to white noise\nPearson CC = '+str(avg)+'\n', fontsize=20)
        plt.title('Cell responses to ' + filename + ' noise\nPearson CC = '+str(avg)+'\n', fontsize=20)

        time = np.array(range(extract_len))
        #plt.imshow(y_show.T, aspect='auto', cmap=cm.gray_r)
        s = 4
        line = 3
        timeshow = extract_len
        for i in range(num_trials):
            spike = np.where(y_show[i, :]>0)
            plt.scatter(time[spike], y_show[i, spike]*i, s=s, marker='|', color='b')
        plt.xlim([0,timeshow])
        plt.ylabel('Data', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(),[])#, ['left','bottom'])

        plt.subplot(3,1,2)
        for i in range(num_trials):
            spike = np.where(y_pre_show[i, :]>0)
            plt.scatter(time[spike], y_pre_show[i, spike]*i, s=s, marker='|', color='r')

        plt.xlim([0,timeshow])
        plt.ylabel('CNN', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), [])

        r_show_max = np.max(r_show)
        r_show = r_show / r_show_max
        #print('r_show_max' + str(r_show_max))

        r_pre_show_max = max(r_pre_show)
        r_pre_show = r_pre_show / r_pre_show_max


        plt.subplot(3,1,3)
        plt.plot(r_show.T * 30, 'b', linewidth=line, alpha=0.4, label='data')
        plt.plot(r_pre_show.T * 30, 'r', linewidth=line, alpha=0.4, label='CNN prediction')
        #plt.legend(loc='upper left')
        plt.ylabel('Rate', fontsize=18)
        plt.xlim([0,timeshow])

        to_xticks = [str(e*2) for e in range(6)]
        plt.xticks(np.array(range(6))*60, to_xticks)

        #plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left','bottom'])

        plt.savefig(self.visualpath + filename + '_response_test_' + self.cell_name + '.png')
        plt.savefig(self.visualpath+ filename + '_response_test.eps')
        #plt.savefig(self.visualpath+'respose_test.svg')
        plt.close()

        return avg


    def draw_test_shuffle_response(self, nstep=10):
        filepath = os.path.join(self.h5path, self.stim_type+'test'+self.toCell+'.hdf5')
        nt = self.nt
        with h5py.File(filepath, 'r') as f:
            X = np.array(f['X'])
            if nt > 1:
                X = rolling_window(X, nt, time_axis=0)
                y = np.array(f['y'])[nt:]
                r = np.array(f['r'])[nt:]
            else:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                y = np.array(f['y'])
                r = np.array(f['r'])
        #print(y.shape)

        metrics = ('cc',)

        extract_len = len(r)
        if nt > 1:
            extract_len = 300-nt

        X_ex = X[:extract_len]

        num_trials = y.shape[1]
        #test_trials = y.shape[1]
        ind_test_trials = np.array(range(y.shape[1]))
        random.shuffle(ind_test_trials)
        y_show = y
        for i in range(y.shape[1]):
            y_show[:,i] = np.array( (y[:,ind_test_trials[i]]) > 0)
        if(num_trials <= y.shape[1]):
            y_show = y_show[:,:num_trials]
        else :
            y_show = y_show[:,:y.shape[1]]


        r_pre = self.model.predict_on_batch(X_ex)[0]
        y_pre = np.zeros((num_trials,)+r_pre.shape)

        for i in range(num_trials):
            y_pre[i] = np.random.poisson(r_pre)
        y_pre_show = np.array(y_pre[:,:extract_len,0]>0)
        r_show = r[:extract_len,0]


        r_pre_show = r_pre[:extract_len,0]
        
        avg_val, all_val = allmetrics(r_show, r_pre_show, metrics)
        avg = avg_val['cc']
        write_txt(self.path +'cc.txt', 'model-'+self.cell_name+'-test-'+self.toCell+'  '+str(avg))

        plt.subplot(3,1,1)
        y_show = y_show.T
        plt.title('Cell responses to white noise\nPearson CC = '+str(avg)+'\n', fontsize=20)

        time = np.array(range(extract_len))
        #plt.imshow(y_show.T, aspect='auto', cmap=cm.gray_r)
        s = 4
        line = 3
        timeshow = 300
        for i in range(num_trials):
            spike = np.where(y_show[i, :]>0)
            plt.scatter(time[spike], y_show[i, spike]*i, s=s, marker='|', color='b')
        plt.xlim([0,timeshow])
        plt.ylabel('Data', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(),[])#, ['left','bottom'])

        plt.subplot(3,1,2)
        for i in range(num_trials):
            spike = np.where(y_pre_show[i, :]>0)
            plt.scatter(time[spike], y_pre_show[i, spike]*i, s=s, marker='|', color='r')

        plt.xlim([0,timeshow])
        plt.ylabel('CNN', fontsize=18)
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), [])


        plt.subplot(3,1,3)
        plt.plot(r_show.T * 30, 'b', linewidth=line, alpha=0.4, label='data')
        plt.plot(r_pre_show.T * 30, 'r', linewidth=line, alpha=0.4, label='CNN prediction')
        #plt.legend(loc='upper left')
        plt.ylabel('Rate', fontsize=18)
        plt.xlim([0,timeshow])

        to_xticks = [str(e*2) for e in range(6)]
        plt.xticks(np.array(range(6))*60, to_xticks)

        #plt.gca().xaxis.set_major_locator(MultipleLocator(2))
        plt.tick_params(axis='y', direction='out')
        plt.tick_params(axis='x', direction='out')
        adjust_spines(plt.gca(), ['left','bottom'])

        plt.savefig(self.visualpath+'respose_test.png')
        plt.savefig(self.visualpath+'respose_test.eps')
        plt.savefig(self.visualpath+'respose_test.svg')
        plt.close()

    def draw_train_STA(self):
        filepath = os.path.join(self.h5path, self.stim_type + 'train.hdf5')
        nt = self.nt

        with h5py.File(filepath, 'r') as f:
            if self.nt > 1:
                X = rolling_window(np.array(f['X']), nt, time_axis=0)
                y = np.array(f['y'])[nt:]
                r = np.array(f['r'])[nt:]
            else:
                X = np.array(f['X'])
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                y = np.array(f['y'])
                r = np.array(f['r'])

            if not self.cfg == None:
                if self.cfg['percent']:
                    num_data = X.shape[0]
                    ex_len = int(np.ceil(num_data*self.cfg['percent_ratio']))
                    X = X[:ex_len]
                    y = y[:ex_len]
                    print('extract len:', ex_len)
                    print('spikes:',np.sum(np.array(y>0)))
                    r = r[:ex_len]
                elif self.cfg['noise']:
                    X = stim_noise(X, self.cfg['noise_ratio'])

        #print(X.shape)
        stim_shape = X.shape[1:]
        samples = X.shape[0]
        batch_size = 1000;

        sta_data = 0
        sta_pre = 0
        for batch in range(int(np.ceil(samples/batch_size))):
            X_batch = X[batch*batch_size : (batch+1)*batch_size]
            y_data = y[batch*batch_size : (batch+1)*batch_size, 0]
            if(batch == int(np.ceil(samples/batch_size))):
                X_batch = X[batch * batch_size:]
                y_data = y[batch * batch_size:, 0]

            len_data = X_batch.shape[0]
            r_pre = self.model.predict_on_batch(X_batch)[0]
            y_pre = np.array(np.random.poisson(r_pre)>0)

            y_flat = y_pre.reshape(len_data, 1)
            X_flat = X_batch.reshape(len_data, -1)
            sta_pre += np.dot(y_flat.T, X_flat)

            y_data = np.array(y_data.reshape(len_data, 1)>0)
            sta_data += np.dot(y_data.T, X_flat)


        sta_data /= samples
        sta_data = sta_data.reshape(stim_shape)
        sta_pre /= samples
        sta_pre = sta_pre.reshape(stim_shape)

        if self.nt > 1:
            spatial_data, temporal_data = ft.decompose(sta_data)
            spatial_pre, temporal_pre = ft.decompose(sta_pre)
        else:
            spatial_data = sta_data[0]
            temporal_data = np.array([])
            spatial_pre = sta_pre[0]
            temporal_pre = np.array([])

        cl = max( np.max(abs(spatial_data)), np.max(abs(spatial_pre)) )
        colorlimit = [-cl, cl]
        plt.subplot(2,2,1)
        plt.title('STA spatial data', fontsize=20)
        plt.imshow(spatial_data, interpolation='nearest', cmap='seismic', clim=colorlimit)
        plt.grid('off')
        plt.xticks([])
        plt.yticks([])
        plt.axis('on')

        plt.subplot(2,2,3)
        plt.title('STA temporal data', fontsize=20)
        plt.plot(np.linspace(0, len(temporal_data) * 10, len(temporal_data)), temporal_data, 'k', linewidth=2)
        plt.grid('off')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])


        plt.subplot(2,2,2)
        plt.title('STA spatial predict', fontsize=20)
        plt.imshow(spatial_pre, interpolation='nearest', cmap='seismic', clim=colorlimit)
        plt.grid('off')
        plt.xticks([])
        plt.yticks([])
        plt.axis('on')

        plt.subplot(2,2,4)
        plt.title('STA temporal predict', fontsize=20)
        plt.plot(np.linspace(0, len(temporal_pre) * 10, len(temporal_pre)), temporal_pre, 'k', linewidth=2)
        plt.grid('off')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])

        plt.savefig(self.visualpath+'sta_train.png')
        plt.savefig(self.visualpath+'sta_train.eps')
        plt.close()

    def get_sta(self, layer_id, samples=5000, batch_size=1000, print_every=None, subunit_id=None, stim_shape = None):
        model = self.model
        if subunit_id is not None:
            activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))
            def get_activations(stim):
                activity = activations(stim)
                # first dimension is batch size
                return activity[:, :, subunit_id[0], subunit_id[1]]
        else:
            get_activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))

        # Initialize STA
        sta = 0
        res = 0

        # Generate white noise and map STA
        for batch in range(int(np.ceil(samples/batch_size))):
            whitenoise = np.random.randn(*((batch_size,) + stim_shape)).astype('float32')
            #whitenoise = rolling_window(whitenoise_2d, self.nt, time_axis=0)
            response = get_activations(whitenoise)
            true_response_shape = response.shape[1:]

            response_flat = response.reshape(batch_size, -1).T
            whitenoise_flat = whitenoise.reshape(batch_size, -1)
            """
            # sta will be matrix of units x sta dims
            """
            sta += np.dot(response_flat, whitenoise_flat)
            #res = get_activations(np.random.randn(*((2,)+stim_shape)).astype('float32'))
            #sta += np.mean(response, axis = 0)

            if print_every:
                if batch % print_every == 0:
                    print('On batch %i of %i...' %(batch, samples/batch_size))
        #print (response.shape, response_flat.shape, whitenoise_flat.shape)
        sta /= samples

        # when the sta is of a conv layer
        if len(true_response_shape) == 3:
            #sta = sta.reshape(true_response_shape[0], true_response_shape[1], true_response_shape[2], -1)
            #sta = sta.reshape(true_response_shape[0], 1,  true_response_shape[1], true_response_shape[2])
            sta = sta.reshape(true_response_shape[0], true_response_shape[1]*true_response_shape[2], -1)
            sta = np.mean(sta, axis=1)
            sta = sta.reshape(*((true_response_shape[0],) + stim_shape))
            print("conv")
            return sta
        # when the sta is of an affine layer
        elif len(true_response_shape) == 1:
            sta = sta.reshape(*((true_response_shape[0],) + stim_shape))
            print("affine")
            return sta
        else:
            print('STA shape not recognized. Returning [sta, shape of response].')
            return [sta, true_response_shape]

    def visualize_sta(self, sta, fig_size=(8, 10), display=True, normalize=True, img_name = ''):

        '''
        if len(sta) == 3:
            num_units = 1
        else:
            num_units = sta.shape[0]
        '''
        print('sta_shape: ' + str(sta.shape))
        num_units = sta.shape[0]

        # plot space and time profiles together
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        num_cols = int(np.sqrt(num_units))
        num_rows = int(np.ceil(num_units/num_cols))
        cl_max = 0
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                if plt_idx > num_units:
                    break
                if self.nt > 1:
                    spatial,temporal = ft.decompose(sta[plt_idx-1])
                else:
                    spatial = sta[plt_idx-1][0]
                cl_max = max(cl_max, np.max(abs(spatial)))
        colorlimit = [-cl_max, cl_max]

        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                if plt_idx > num_units:
                    break
                if self.nt > 1:
                    spatial,temporal = ft.decompose(sta[plt_idx-1])
                else:
                    spatial = sta[plt_idx-1][0]
                    temporal = np.array([])

                ax = plt.subplot2grid((num_rows*4, num_cols), (4*y, x), rowspan=3)
                ax.imshow(matrix_rotate(spatial), interpolation='nearest', cmap='seismic', clim=colorlimit)
                #plt.title('Feature %i' % plt_idx)
                plt.grid('off')
                plt.xticks([])
                plt.yticks([])
                plt.axis('on')

                ax = plt.subplot2grid((num_rows * 4, num_cols), (4 * y + 3, x), rowspan=1)
                ax.plot(np.linspace(0, len(temporal) * 10, len(temporal)), temporal, 'k', linewidth=2)
                plt.grid('off')
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])

        plt.suptitle('Iteration ' + str(self.iters))
        plt.savefig(self.visualpath + img_name + str(self.iters)  +'.png', dpi=300)
        plt.savefig(self.visualpath + img_name + '.eps', dpi=300)
        plt.close()
        if display:
            plt.show()

    def visualize_response(self, res, fig_size=(8, 10), img_name = ''):

        num_units = res.shape[1]
        colorlimit = [-np.max(abs(res)), np.max(abs(res))]

        # plot space and time profiles together
        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title('Response', fontsize=20)
        num_cols = int(np.sqrt(num_units))
        num_rows = int(np.ceil(num_units/num_cols))
        for x in range(num_cols):
            for y in range(num_rows):
                plt_idx = y * num_cols + x + 1
                if plt_idx > num_units:
                    break

                spatial = res[0][plt_idx-1]
                ax = plt.subplot2grid((num_rows*4, num_cols), (4*y, x), rowspan=3)
                ax.imshow(spatial, interpolation='nearest', cmap='seismic', clim=colorlimit)
                plt.title('Response %i' % plt_idx)
                plt.grid('off')
                plt.xticks([])
                plt.yticks([])
                plt.axis('on')

        plt.savefig(self.visualpath + img_name + '.png', dpi=300)
        plt.savefig(self.visualpath + img_name + '.eps', dpi=300)
        plt.close()

    def animation_feature(self, layer_id=0, num_iters = 5, samples=5000, batch_size=1000, fig_size=(8,10), stim_shape=None):

        print ('****************Animation Features****************')

        interval = 500
        sta_list = []
        for i in range(num_iters-1):
            model = self.read_model(iters= i*interval)
            get_activations = theano.function([model.layers[0].input], model.layers[layer_id].get_output(train=False))

            # Initialize STA
            sta = 0

            # Generate white noise and map STA
            for batch in range(int(np.ceil(samples / batch_size))):
                whitenoise = np.random.randn(*((batch_size,) + stim_shape)).astype('float32')
                response = get_activations(whitenoise)
                true_response_shape = response.shape[1:]

                response_flat = response.reshape(batch_size, -1).T
                whitenoise_flat = whitenoise.reshape(batch_size, -1)
                """
                # sta will be matrix of units x sta dims
                """
                sta += np.dot(response_flat, whitenoise_flat)
            sta /= samples

            sta = sta.reshape(true_response_shape[0], true_response_shape[1] * true_response_shape[2], -1)
            sta = np.mean(sta, axis=1)
            sta = sta.reshape(*((true_response_shape[0],) + stim_shape))
            sta_list.append(sta)

        sta_max = np.max(abs(np.array(sta_list)))
        colorlimit = [-sta_max, sta_max]

        fig = plt.gcf()
        fig.set_size_inches(fig_size)
        plt.title('STA', fontsize=20)
        num_units = sta_list[0].shape[0]
        num_cols = int(np.sqrt(num_units))
        num_rows = int(np.ceil(num_units/num_cols))
        def make_frame(t):
            global iters_f
            #plt.title('Iters '+str(iters_f), fontsize=20)
            if (iters_f>=0 and iters_f < num_iters-1):
                print ('Animate Features',iters_f)
                for x in range(num_cols):
                    for y in range(num_rows):
                        plt_idx = y * num_cols + x + 1
                        if plt_idx > num_units:
                            break
                        spatial = sta_list[iters_f][plt_idx - 1][0]
                        ax = plt.subplot2grid((num_rows * 4, num_cols), (4 * y, x), rowspan=3)
                        ax.imshow(matrix_rotate(spatial), interpolation='nearest', cmap='seismic', clim=colorlimit)
                        plt.title('Feature %i' % plt_idx)
                        plt.grid('off')
                        plt.xticks([])
                        plt.yticks([])
                        plt.axis('on')
                space = np.zeros_like(spatial)
                ax = plt.subplot2grid((num_rows * 4, num_cols), (4 * y, x), rowspan=3)
                ax.imshow(space, interpolation='nearest', cmap='seismic', clim=colorlimit)
                plt.grid('off')
                plt.axis('off')
                iters_show = iters_f*interval
                plt.text(-70, 15, 'Iteration %i'%iters_show, fontdict={'size': 24, 'color': 'black'})
            iters_f += 1
            return mplfig_to_npimage(fig)

        fps = 5
        duration = num_iters/fps
        animation = VideoClip(make_frame, duration=duration)
        animation.write_gif(self.visualpath + 'layer_' + str(layer_id) + '_features_film.gif', fps=fps)


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def rolling_window(array, window, time_axis=0):

    if time_axis == 0:
        array = array.T

    elif time_axis == -1:
        pass

    else:
        raise ValueError('Time axis must be 0 (first dimension) or -1 (last)')

    assert window >= 1, "`window` must be at least 1."
    assert window < array.shape[-1], "`window` is too long."

    # with strides
    shape = array.shape[:-1] + (array.shape[-1] - window, window)
    strides = array.strides + (array.strides[-1],)
    arr = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)

    if time_axis == 0:
        return np.rollaxis(arr.T, 1, 0)
    else:
        return arr

def matrix_rotate(A):
    h = A.shape[0]
    w = A.shape[1]
    Ar = np.zeros_like(A)
    for i in range(h):
        for j in range(w):
            Ar[i,j] = A[h-1-i,w-1-j]
    return Ar

def write_txt(path, data):
    with open(path, 'a') as f:
        f.write(data + '\n')

def stim_noise(X, ratio):
    X_flat = X.reshape(X.shape[0], -1)
    print('a')
    import sys
    sys.setrecursionlimit(10000000)
    for i in range(len(X_flat)):
        print(i)
        pixels = list(np.arange(len(X_flat[i])))
        import random
        pixels_select = random.sample(pixels, int(ratio * len(X_flat[i])))
        import copy
        pixels_shuffle = copy.deepcopy(pixels_select)
        random.shuffle(pixels_shuffle)
        X_flat[i][pixels_select] = X_flat[i][pixels_shuffle]
    print('b')
    return X_flat.reshape(X.shape)
