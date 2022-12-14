'''Implements pruning class.'''  

import os
import shutil
import numpy as np
import tensorflow as tf
try:
    from models import *
    from helper import *
    from trainer import iterPruning, train_one_step, test_one_step
except:
    pass


class pruning(object):

    '''Sets up a pruning experiment.'''  

    def __init__(self, ds_train, ds_test, 
                model_params,
                train_params,
                epochs_for_pruning = 10,
                num_pruning = 10,
                step_perc = 0.5,
                verbose = True,
                same_init = False):
        '''
        Args:   
            - ds_train: dataset for training, already batched
            - ds_test: dataset for testing, already batched
            - model_params: dictionary of parameters
                            contains the following entries   
                            - 'layers': list of hidden layers sizes
                            - 'initializer': initializer for weights
                            - 'activation': activation function, default relu
                            - 'final_activation': activation function for final layer, default None
                            - 'BatchNorm': boolean, whether to use batch normalization, default False
                            - 'Dropout': list of dropout rates, default None
                            - 'optimizer': optimizer for training, default Adam
                            - 'loss_fn': loss function, default SparseCategoricalCrossentropy
                            - 'metrics': metrics for training, default SparseCategoricalAccuracy
            - train_params: dictionary of parameters for training
                            contains the following entries
                            - 'train_loss': loss object, default Mean
                            - 'train_acc': accuracy object, default SparseCategoricalAccuracy
            - epochs: epochs to train the model before pruning
            - num_pruning: no. of rounds to prune
            - step_perc: percentage to prune
            - verbose: boolean, whether to print out information, default True  
            - same_init: boolean, whether to use the same initial weights after pruning, default False
        '''
        self.ds_train = ds_train
        self.ds_test = ds_test
        self.model_params = model_params
        self.train_params = train_params
        self.epochs_for_pruning = epochs_for_pruning
        self.num_pruning = num_pruning
        self.step_perc = step_perc
        self.verbose = verbose    
        self.same_init = same_init

    @property
    def batch_size(self):
        for img,label in self.ds_train:
            num = img.shape[0]
        return num

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
        Iteratively prunes the model and returns the initial masks.
        '''  
        return iterPruning(self.makeModel,
                    self.ds_train,
                    self.ds_test,
                    self.model_params,
                    self.train_params,
                    epochs = self.epochs_for_pruning,
                    num_pruning=self.num_pruning,
                    step_perc=self.step_perc,
                    verbose=self.verbose,
                    same_init=self.same_init)

    def test_model(self):
        '''Checks model is built correctly and returns trained model.'''  
        model = self.makeModel()
        model.summary()
        model.compile(self.model_params['optimizer'],
                    self.model_params['loss_fn'],
                    self.model_params['metrics'])
        model.fit(self.ds_train,
                batch_size = self.batch_size,
                epochs = self.epochs_for_pruning,
                validation_data = self.ds_test)  
        return model


    def test_training(self, epochs = 5):
        '''
        Checks training loop is written correctly.   

        Args:
            - epochs: int, number of epochs to test training, default 5 
        
        '''  
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

    #TODO: add test for evaluating model performance
    # def evaluate(self):




    def removeLogs(self):
        """Removes logs from previous runs."""
        for x in os.scandir('logs'):
            if x.name != '.gitkeep':
                print("Removing " + x.name)
                shutil.rmtree(x.path)

    def removeModels(self):
        """Removes models from previous runs."""
        for x in os.scandir('saved_models'):
            if x.name != '.gitkeep':
                print("Removing " + x.name)
                shutil.rmtree(x.path)  

    



        


    
    
    
    


