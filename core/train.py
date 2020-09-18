"""
Core tools for training models
"""
from keras.models import Model
from .metrics import allmetrics
from time import time
import tableprint as tp
import copy
import numpy as np

__all__ = ['train']


def train(model, data, monitor, num_iters, num_epochs, begin_iters=0):
    """Train the given network against the given data

    Parameters
    ----------
    model : keras.models.Model or glms.GLM
        A GLM or Keras Model object

    experiment : experiments.Experiment
        An Experiment object

    monitor : io.Monitor
        Saves the model parameters and plots of performance progress

    num_epochs : int
        Number of epochs to train for

    reduce_lr_every : int
        How often to reduce the learning rate

    reduce_rate : float
        A fraction (constant) to multiply the learning rate by

    """
    assert isinstance(model, Model), "'model' must be a Keras model"

    # initialize training iteration
    iteration = begin_iters
    train_start = time()
    loss_list = []
    avg_list = []
    all_list = []
    iteration_list = []
    max_val = 0
    stop = False
    max_model = None

    import sys
    sys.setrecursionlimit(10000000)

    # loop over epochs
    try:
        for epoch in range(num_epochs):
            tp.banner('Epoch #{} of {}'.format(epoch + 1, num_epochs))
            print(tp.header(["Iteration", "Loss", "Runtime"]), flush=True)
            
            # loop over data batches for this epoch 
            for X, y in data.train(shuffle=True):#np.random.randint(0,3)
                #print('loop over data batches for this epoch')
                # train on the batch

                tstart = time()
                loss = model.train_on_batch(X, y)[0]
                
                #validate on the batch
                if(iteration % 20 == 0):
                #if(iteration % 1 == 0 and iteration>=0):
                    avg_val, all_val = validate(model, data)
                    avg_list.append(avg_val)
                    all_list.append(all_val)
                    iteration_list.append(iteration)
                    loss_list.append(loss)
                    print(tp.header(["Iteration", "Average_Val"]), flush=True)
                    print(tp.row([iteration, float(avg_val['cc'])]), flush=True)
                    print(tp.bottom(2))
                    
                    print(tp.header(["Iteration", "max_val"]), flush=True)
                    print(tp.row([iteration, max_val]), flush=True)
                    print(tp.bottom(2))

                    #if iteration > 10000:
                    #    max_val = 0.55
                    if float(avg_val['cc']) > max_val + 0.01 and iteration >= 500:

                        print('cc value bigger than max val..')
                        max_val = float(avg_val['cc'])
                        del max_model
                        import gc
                        gc.collect()
                        max_model = copy.deepcopy(model)

                    if iteration >= num_iters:
                        stop = True

                    
                elapsed_time = time() - tstart

                if(iteration % 2000 == 0):
                    monitor.save_model(epoch, iteration, model)

                # update
                iteration += 1
                print(tp.row([iteration, float(loss), tp.humantime(elapsed_time)]), flush=True)

                if stop == True:
                    break;

            print(tp.bottom(3))

            if stop == True:
                break;


    except KeyboardInterrupt:
        print('\nCleaning up')

    tp.banner('Training complete!')

    try:
        monitor.save_model(epoch, iteration, max_model, is_final=True)

    except AttributeError:
        tp.banner('Final model is None object!')
        tp.banner('Copying from the old model..')
        max_model = copy.copy(model)
        monitor.save_model(epoch, iteration, max_model, is_final=True)

    monitor.save_performance(iteration_list, loss_list, avg_list, all_list)

    return iteration
    
def validate(model, data):
    X_val,y_val = data.validate()
    y_val_true = y_val

    y_out = model.predict_on_batch(X_val)[0]
    y_val_pre = y_out
    
    metrics = ('cc',)

    avg_val, all_val = allmetrics(y_val_true, y_val_pre, metrics)
    return avg_val, all_val
