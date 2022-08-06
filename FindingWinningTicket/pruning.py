'''Implements pruning class.'''  

import os
import shutil
import numpy as np
import tensorflow as tf
from models import *
from helper import *
from trainer import iterPruning, train_one_step, test_one_step


class pruning(object):

    '''Sets up a pruning experiment.'''  

    def __init__(self, ds_train, ds_test, 
                model_params,
                train_params,
                epochs_for_pruning = 10,
                num_pruning = 10,
                step_perc = 0.5):
        '''
        Args:   
            - ds_train: dataset for training, already batched
            - ds_test: dataset for testing, already batched
            - model_params: dictionary of parameters
                            contains the following entries  
                            - 'layers'
                            - 'initializer'
                            - 'activation'
                            - 'BatchNorm'
                            - 'Dropout'
                            - 'optimizer'
                            - 'loss_fn'
                            - 'metrics'
            - train_params: dictionary of parameters for training
                            contains the following entries
                            - 'train_loss': 
                            - 'train_acc':
            - epochs: epochs to train the model before pruning
            - num_pruning: no. of rounds to prune
            - step_perc: percentage to prune
        '''
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model_params = model_params
        self.train_params = train_params
        self.epochs_for_pruning = epochs_for_pruning
        self.num_pruning = num_pruning
        self.step_perc = step_perc    

    @property
    def batch_size(self):
        for img,label in self.ds_train:
            num = img.shape[0]
        return num

    # TODO: add Dataset implementaion
    
    # @property
    # def dset(self):
        # some processing goes in here
        # to modify user input
        # pass
    # @dset.setter

    # @property
    # def ds_train(self):
    #     pass

    # @property
    # def ds_test(self):
    #     pass
    

    def makeModel(self, preinit_weights = None, masks = None):
        '''
        Initializes model before pruning.
        '''
        return makeFC(
                    preinit_weights,
                    masks,
                    **self.model_params)
    
    def prune(self):
        '''
        Iteratively prunes the model.
        '''  
        iterPruning(self.makeModel,
                    self.ds_train,
                    self.ds_test,
                    self.model_params,
                    self.train_params,
                    epochs = self.epochs_for_pruning,
                    num_pruning=self.num_pruning,
                    step_perc=self.step_perc)  

    def test_model(self):
        '''Checks model is built correctly.'''  
        model = self.makeModel()
        model.summary()
        model.compile(self.model_params['optimizer'],
                    self.model_params['loss_fn'],
                    self.model_params['metrics'])
        model.fit(self.ds_train,
                batch_size = self.batch_size,
                epochs = self.epochs_for_pruning,
                validation_data = self.ds_test)  


    def test_training(self, epochs = 5):
        '''Checks training loop is written correctly.'''  
        model = makeFC()
        fake_mask = test_masks(model)
        model.summary()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        optimizer = tf.keras.optimizers.Adam(0.001)

        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

        test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')
        original_acc = []

        for epoch in range(epochs):  

            print(f"Epoch {epoch} for original")
            train_loss.reset_states()
            train_acc.reset_states()
            
            for x,y in self.ds_train:
                train_one_step(model, x,y, 
                            fake_mask, optimizer,
                            loss_fn, train_acc, train_loss)

            # optional printing of training accuracy
            print(f"OG train acc {train_acc.result()}")

            for x,y in self.ds_test:
                test_one_step(model, x, y, loss_fn,
                            test_acc, test_loss)

            print(f"Original model accuracy {test_acc.result()} \n")
            original_acc.append(test_acc.result())

            test_loss.reset_states()
            test_acc.reset_states()

        original_best = np.max(np.array(original_acc))
        print(f"\n Original model training finished. Highest Accuracy {original_best} \n")




    def removeLogs(self):
        """Removes logs from previous runs."""
        for x in os.scandir('.\logs'):
            if x.name != '.gitkeep':
                print("Removing " + x.name)
                shutil.rmtree(x.path)

    def removeModels(self):
        """Removes models from previous runs."""
        for x in os.scandir('.\saved_models'):
            if x.name != '.gitkeep':
                print("Removing " + x.name)
                shutil.rmtree(x.path)  

    



        


    
    
    
    


