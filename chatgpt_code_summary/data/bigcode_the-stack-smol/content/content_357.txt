"""
DeepChEmbed （DCE) Models
"""
from dimreducer import DeepAutoEncoder
from cluster import KMeansLayer
from cluster import KMeans
from keras import Model
from keras import optimizers
from keras.utils import normalize
import numpy as np

class DCE():
    """
    The class to build a deep chemical embedding model.

    Attributes:
        autoencoder_dims: a list of dimensions for encoder, the first
                          element as input dimension, and the last one as
                          hidden layer dimension.
        n_clusters: int, number of clusters for clustering layer.
        alpha: float, parameters for soft label assigning.
        update_interval: int, indicating every number of epoches, the harhened
                         labels will be upadated and/or convergence cretia will
                         be examed.
        max_iteration: int, maximum iteration for the combined training
        clustering_tol: float, convergence cretia for clustering layer
        model: keras Model variable
        HARDENING_FUNCS: smoothsetp hardening functions for unsupervised DCE
                         training, up to 9th order
    """

    HARDENING_FUNCS = {
        1: lambda x: x,
        3: lambda x: (-2*x + 3) * x**2,
        5: lambda x: ((6*x - 15)*x + 10) * x**3,
        7: lambda x: (((-20*x + 70)*x - 84)*x + 35) * x**4,
        9: lambda x: ((((70*x - 315)*x + 540)*x -420)*x + 126) * x**5}

    def __init__(self, autoencoder_dims, n_clusters, update_interval=50,
                 max_iteration=1e4, clustering_tol=1e-4, alpha=1.0):
        """Construtor of DCE. """
        self.autoencoder_dims = autoencoder_dims
        self.n_clusters       = n_clusters
        self.alpha            = alpha
        self.update_interval  = update_interval
        self.max_iteration    = max_iteration
        self.clustering_tol   = clustering_tol
        self.model            = None

        return

    def build_model(self, norm=True, act='relu'):
        """Build DCE using the initialized attributes

        Args:
            norm: boolean, wheher to add a normalization layer at the begining
                  of the autoencoder
            act: string, keras activation function name for autoencoder
        """
        autoencoder = DeepAutoEncoder(self.autoencoder_dims, act)
        autoencoder.build_model(norm=norm)
        embeding = autoencoder.model.get_layer(name='embedding_layer').output
        clustering = KMeansLayer(self.n_clusters, alpha=self.alpha,
                                 name='clustering')(embeding)
        self.model = Model(inputs=autoencoder.model.input,
                           outputs=[clustering,autoencoder.model.output])

        return

    def train_model(self, data_train,
                    labels_train=None, data_test=None, labels_test=None,
                    verbose=1,
                    compiled=False, clustering_loss='kld',
                    decoder_loss='mse',clustering_loss_weight=0.5,
                    hardening_order=1, hardening_strength=2.0,
                    compiled=False,
                    optimizer='adam', lr=0.001, decay=0.0):
        """Train DCE Model:

            If labels_train are not present, train DCE model in a unsupervised
        learning process; otherwise, train DCE model in a supervised learning
        process.

        Args:
            data_train: input training data
            labels_train: true labels of traning data
            data_test: input test data
            labels_test: true lables of testing data
            verbose: 0, turn off the screen prints
            clustering_loss: string, clustering layer loss function
            decoder_loss:, string, decoder loss function
            clustering_loss_weight: float in [0,1], w_c,
            harderning_order: odd int, the order of hardening function
            harderning_strength: float >=1.0, the streng of the harderning
            compiled: boolean, indicating if the model is compiled or not
            optmizer: string, keras optimizers
            lr: learning rate
            dacay: learning rate dacay

        Returns:
            train_loss:  training loss
            test_loss: only if data_test and labels_test are not None in
                       supervised learning process
        """
        if (not compiled):
            assert clustering_loss_weight <= 1 and clustering_loss_weight >= 0

            if optimizer == 'adam':
                dce_optimizer = optimizers.Adam(lr=lr,decay=decay)
            elif optimizer == 'sgd':
                dce_optimizer = optimizers.sgd(lr=lr,decay=decay)
            else:
                raise Exception('Input optimizer was not found')

            self.model.compile(loss={'clustering': clustering_loss,
                                     'decoder_output': decoder_loss},
                               loss_weights=[clustering_loss_weight,
                                             1 - clustering_loss_weight],
                               optimizer=dce_optimizer)

        if (labels_train is not None):
            supervised_learning = True
            if verbose >= 1: print('Starting supervised learning')
        else:
            supervised_learning = False
            if verbose >= 1: print('Starting unsupervised learning')

        # initializing model by using sklean-Kmeans as guess
        kmeans_init = KMeans(n_clusters=self.n_clusters)
        kmeans_init.build_model()
        encoder  = Model(inputs=self.model.input,
                         outputs=self.model.get_layer(\
                         name='embedding_layer').output)
        kmeans_init.model.fit(encoder.predict(data_train))
        y_pred_last = kmeans_init.model.labels_
        self.model.get_layer(name='clustering').\
            set_weights([kmeans_init.model.cluster_centers_])

        # Prepare training: p disctribution methods
        if not supervised_learning:
            # Unsupervised Learning
            assert hardening_order in DCE.HARDENING_FUNCS.keys()
            assert hardening_strength >= 1.0
            h_func = DCE.HARDENING_FUNCS[hardening_order]
        else:
            # Supervised Learning
            assert len(labels_train) == len(data_train)
            assert len(np.unique(labels_train)) == self.n_clusters
            p = np.zeros(shape=(len(labels_train), self.n_clusters))
            for i in range(len(labels_train)):
                p[i][labels_train[i]] = 1.0

            if data_test is not None:
                assert len(labels_test) == len(data_test)
                assert len(np.unique(labels_test)) == self.n_clusters
                p_test = np.zeros(shape=(len(labels_test), self.n_clusters))
                for i in range(len(labels_test)):
                    p_test[i][labels_test[i]] = 1.0

                validation_loss = []

        # training start:
        loss = []

        for iteration in range(int(self.max_iteration)):

            if iteration % self.update_interval == 0:
                # updating p for unsupervised learning process
                q, _ = self.model.predict(data_train)
                if not supervised_learning:
                    p = DCE.hardening(q, h_func, hardening_strength)

                # get label change i
                y_pred = q.argmax(1)
                delta_label_i = np.sum(y_pred != y_pred_last).\
                    astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred

                # exam convergence
                if iteration > 0 and delta_label_i < self.clustering_tol:
                    print(str(delta_label_i) +' < ' + str(self.clustering_tol))
                    print('Reached tolerance threshold. Stopping training.')
                    break

            loss.append(self.model.train_on_batch(x=data_train,
                                                  y=[p,data_train]))
            if supervised_learning and data_test is not None:
                validation_loss.append(self.model.test_on_batch(
                    x=data_test, y=[p_test,data_test]))

            if verbose > 0 and iteration % self.update_interval == 0:
                print('Epoch: ' + str(iteration))
                if verbose == 1:
                    print('  Total_loss = ' + str(loss[iteration][0]) +
                          ';Delta_label = ' + str(delta_label_i))
                    print('  Clustering_loss = ' + str(loss[iteration][1]) +
                          '; Decoder_loss = ' + str(loss[iteration][2]))

        if iteration == self.max_iteration - 1:
            print('Reached maximum iteration. Stopping training.')

        if data_test is None:
            return np.array(loss).T
        else:
            return [np.array(loss).T, np.array(validation_loss).T]

    @staticmethod
    def hardening(q, h_func, stength):
        """hardening distribution P and return Q

        Args:
            q: input distributions.
            h_func: input harderning function.
            strength: hardening strength.

        returns:
            p: hardened and normatlized distributions.

        """
        q = h_func(q)
        weight =  q ** stength / q.sum(0)
        return (weight.T / weight.sum(1)).T
