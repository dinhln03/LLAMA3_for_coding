"""Polynomial model class used by agents for building stuff.
"""
from torch import nn, optim

import torch
import torch.nn.functional as F

from stock_trading_backend.agent.model import Model


class NNModel(nn.Module):
    """Torch neural network model.
    """
    def __init__(self, num_inputs, num_hidden_layers, num_inner_features):
        """Initializer for linear model.

        Args:
            num_inputs: the dimension of input data.
            num_hidden_layers: the number of hidden layers.
            num_inner_features: the number of features in the hidden layers
        """
        super(NNModel, self).__init__()
        self.input_layer = nn.Linear(num_inputs, num_inner_features)
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(nn.Linear(num_inner_features, num_inner_features))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(num_inner_features, 1)

    def forward(self, input_tensor):
        """Forward pass on the neural network model.

        Args:
            input_tensor: the input tensor.

        Returns:
            Tensor with model results.
        """
        output = F.relu(self.input_layer(input_tensor))
        output = self.hidden_layers(output)
        output = self.output_layer(output)
        return output


class NeuralNetworkModel(Model):
    """Neural netowrk model class.
    """
    name = "neural_network_model"

    def __init__(self, learning_rate=1e-3, num_hidden_layers=1, num_inner_features=100):
        """Initializer for model class.

        Args:
            learning_rate: the learning rate of the model.
            num_hidden_layers: number of hidden layers in the network.
            num_inner_features: number of features in the hidden layers.
        """
        super(NeuralNetworkModel, self).__init__()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_inner_features = num_inner_features
        self.id_str = "{}_{}_{}_{}".format(self.name, learning_rate, num_hidden_layers,
                                           num_inner_features)

    def _init_model(self, num_inputs):
        """Initializes internal linear model.

        Args:
            num_inputs: number of inputs that model will have.
        """
        self.model = NNModel(num_inputs, self.num_hidden_layers, self.num_inner_features)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _predict(self, state_action_tensor):
        """Use provided information to make a prediction.

        Args:
            state_action_tensor: pytorch tensor with state-action values.

        Returns:
            Predicted values for observation-action tensors.
        """
        if self.model is None:
            self._init_model(state_action_tensor.shape[1])
        return self.model(state_action_tensor).detach().reshape(-1)

    def _train(self, state_action_tensor, expected_values_tensor):
        """Train the model for 1 epoch.

        Args:
            state_action_tensor: pytorch tensor with state-action expected_values.
            expected_values: pytorch tensor with expected values for each state-action.

        Returns:
            The loss before trainig.
        """
        if self.model is None:
            self._init_model(state_action_tensor.shape[1])

        self.optimizer.zero_grad()
        output = self.model(state_action_tensor)
        loss = self.criterion(output, expected_values_tensor)
        loss_value = loss.data.item()
        loss.backward()
        self.optimizer.step()
        return loss_value
