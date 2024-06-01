from engine.steps.IStep import IStep
from keras.models import Model
from keras import backend as K


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

class config_model(IStep):
    """config model"""

    create_Optimizer_func = None
    create_loss_func = None

    def __init__(self, output_channel, name, create_Optimizer_func, create_loss_func):
        super().__init__(self, output_channel, name)
        self.create_Optimizer_func = create_Optimizer_func
        self.create_loss_func = create_loss_func

    def IRun(self):
        if self.create_Optimizer_func == None:
            raise Exception( "No create optimizer function!" )
       
        if self.create_loss_func == None:
            self.create_loss_func = self._default_categorical_crossentropy

        try:        
            opt = self.create_Optimizer_func(self)
            loss = self.create_loss_func(self)
            model = self.output_channel['model']

            """
            if self.train_only_top_layer:
                for layer in base_model.layers:
                    layer.trainable = False
            """

            model.compile(optimizer=opt, loss=loss, metrics=[self.metrics] )
            

        except Exception as e:
            self.output_channel['Error'] = "fatal error occur: " + e.message
            self.output_channel['ErrorType'] = "fatal"

    def IParseConfig( self, config_json ):
        self.epochs = config_json['epochs']
        self.learning_ratio = config_json['learning_ratio']
        self.batch_size = config_json['batch_size']
        self.metrics = config_json['metrics']
     
        self.output_channel['epochs'] = self.epochs
        self.output_channel['learning_ratio'] = self.learning_ratio
        self.output_channel['batch_size'] = self.batch_size

    def IDispose( self ):
        pass
        
    def _default_categorical_crossentropy():
        return "categorical_crossentropy"


class config_model_adam_categorical_crossentropy(config_model):
    """ config model: optimizer=Adam, loss = 'categorical_crossentropy' """

    def __init__(self, output_channel, name=None ):
        super().__init__(self, output_channel, name, self.create_Adam, self.create_loss )

    def create_Adam( self ):        
        return Adam(lr=self.learning_ratio, decay=self.learning_ratio / self.epochs )

    def create_loss( self ):        
        """ create loss function """
        return "categorical_crossentropy"

