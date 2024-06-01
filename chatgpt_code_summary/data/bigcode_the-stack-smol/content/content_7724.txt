import tensorflow as tf 
from tensorflow.keras.callbacks import ModelCheckpoint

import os 


class TensorBoardFix(tf.keras.callbacks.TensorBoard):
    """
    This fixes incorrect step values when using the TensorBoard callback with custom summary ops
    https://stackoverflow.com/questions/64642944/steps-of-tf-summary-operations-in-tensorboard-are-always-0
    """
    def on_train_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_train_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._train_step)
        
    def on_test_begin(self, *args, **kwargs):
        super(TensorBoardFix, self).on_test_begin(*args, **kwargs)
        tf.summary.experimental.set_step(self._val_step)



def get_callbacks(model_name='model',root_dir='logs/fit/',
                  monitor='val_categorical_accuracy',mode='max',
                  save_freq='epoch',save_best_only=True,
                  ):
    log_dir = os.path.join(root_dir,model_name)

    tensorboard = TensorBoardFix(log_dir=log_dir,
                                 histogram_freq=1,
                                 update_freq=50,
                                 )
    
    save_model = ModelCheckpoint(filepath=os.path.join(log_dir,'model.h5'),
                            save_weights_only=False,
                            monitor=monitor,
                            mode=mode,
                            save_best_only=save_best_only,
                            save_freq=save_freq)
    
    return [tensorboard,save_model]