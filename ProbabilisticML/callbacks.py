import numpy as np
import tensorflow as tf  

class PrintProgress(tf.keras.callbacks.Callback):
    
    def __init__(self, num_epochs, **kwargs):
        """
        Initializes the PrintProgress callback.

        Args:
            - num_epochs: int, every num_epochs epochs the progress is printed
        """
        super(PrintProgress, self).__init__(**kwargs)
        self.num_epochs = num_epochs

    def on_epoch_end(self, epoch, logs= None):
        # for key in logs.keys():
        #     print(key)
        train_loss = logs['loss']
        train_acc  = logs['accuracy']
        val_acc   = logs['val_accuracy']
        val_loss   = logs['val_loss']
        if epoch>0 and (epoch+1)%(self.num_epochs)==0:
          print("Epoch {:0} train loss is {:.4f}, train accuracy is {:.4f}, val loss is {:.4f}, and val accuracy is {:.4f}".format(epoch+1, train_loss, train_acc, val_loss, val_acc))

        
class CheckpointCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, directory):
        """
        Initializes the CheckpointCallback.  

        Args:
            - directory: string, path to the directory where the checkpoint is saved
        """
        super(CheckpointCallback, self).__init__()
        self.directory = directory
        self.best_val = tf.Variable(np.inf, trainable=False)
        
    def set_model(self, model):
        self.model = model
        self.ckpt = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.directory, 
                                                  checkpoint_name='model', max_to_keep=1)
        
    def on_epoch_end(self, epoch, logs=None):
        val = logs['val_loss']
        if val < self.best_val:
            self.best_val = val
            self.manager.save()      

