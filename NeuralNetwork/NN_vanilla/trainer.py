import tensorflow as tf


def compile_and_fit(model, optimizer, loss, num_epochs, train_dataset, 
                    validation_dataset=None, metrics=None, callbacks=None,
                    verbose=False):
    """
    Returns history of training.  

    Args:
        - model: tf.keras.Model, model to be trained
        - optimizer: tf.keras.optimizers.Optimizer, optimizer to be used for training
        - loss: tf.keras.losses, loss function to be used for training
        - num_epochs: int, number of epochs to be trained
        - train_dataset: tf.data.Dataset, dataset for training
        - validation_dataset: tf.data.Dataset, dataset for validation
        - metrics: tf.keras.metrics, metrics to be used for evaluation
        - callbacks: list of tf.keras.callbacks.Callback, callbacks to be used for training  
        - verbose: bool, specifies if training progress is printed to stdout
    """  
    
    model.compile(optimizer, loss, metrics)
    history = model.fit(train_dataset, validation_data=validation_dataset,
              epochs = num_epochs, callbacks = callbacks, verbose=verbose)
    return history