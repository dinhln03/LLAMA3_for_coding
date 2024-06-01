
from typing import Tuple

import torch as th
import torch.nn as nn
from torchvision import transforms

from autoencoding_rl.latent_extractors.autoencoder.SimpleEncoder import SimpleEncoder
from autoencoding_rl.latent_extractors.autoencoder.SimpleDecoder import SimpleDecoder
from autoencoding_rl.utils import Transition

class DynAutoencoder(nn.Module):
    
    def __init__(self,  observation_width: int,
                        observation_height: int,
                        observation_channels_num: int,
                        dyn_encoding_size: int,
                        static_encoding_size: int,
                        action_size: int,
                        dynamics_nn_arch: Tuple[int, int]):

        super().__init__()

        self._observation_height = observation_height
        self._observation_width = observation_width
        self._dyn_encoding_size = dyn_encoding_size
        self._static_encoding_size = static_encoding_size
        self._action_size = action_size
        self._observation_channels_num = observation_channels_num
        self._dynamics_nn_arch = dynamics_nn_arch

        self._dynEncoder = SimpleEncoder(encoding_size = self._dyn_encoding_size,
                                         image_channels_num = self._observation_channels_num,
                                         net_input_width = self._observation_width,
                                         net_input_height = self._observation_height)
        
        if self._static_encoding_size != 0:
            self._staticEncoder = SimpleEncoder(encoding_size = self._static_encoding_size,
                                                image_channels_num = self._observation_channels_num,
                                                net_input_width = self._observation_width,
                                                net_input_height = self._observation_height)
        else:
            self._staticEncoder = None

        self._dynamics_net = th.nn.Sequential(  th.nn.Linear(self._dyn_encoding_size+self._action_size, self._dynamics_nn_arch[0]),
                                                th.nn.ReLU(),
                                                th.nn.Linear(self._dynamics_nn_arch[0], self._dynamics_nn_arch[1]),
                                                th.nn.ReLU(),
                                                th.nn.Linear(self._dynamics_nn_arch[1], self._dyn_encoding_size+1))

        self._decoder = SimpleDecoder(  encoding_size = self._dyn_encoding_size + self._static_encoding_size,
                                        image_channels_num = self._observation_channels_num,
                                        net_output_width = self._observation_width,
                                        net_output_height = self._observation_height)

        self._resizeToInput = transforms.Resize((self._observation_height,self._observation_width))


    @property
    def observation_height(self):
        return self._observation_height

    @property
    def observation_width(self):
        return self._observation_width

    @property
    def dyn_encoding_size(self):
        return self._dyn_encoding_size

    @property
    def static_encoding_size(self):
        return self._static_encoding_size

    @property 
    def action_size(self):
        return self._action_size

    def forward(self, transition_batch : Transition):
        observation_batch = transition_batch.observation
        action_batch = transition_batch.action
        assert action_batch.size()[0] == observation_batch.size()[0], \
                f"Observation batch and action batch should have the same length. Action batch size = {action_batch.size()[0]}, observation batch size = {observation_batch.size()[0]}. Action tensor size = {action_batch.size()[0]}. Observation tensor size = {observation_batch.size()[0]}"
        assert observation_batch.size() == (observation_batch.size()[0], self._observation_channels_num, self._observation_height, self._observation_width), \
                f"Observation size should be (Any, {self._observation_channels_num}, {self._observation_height}, {self._observation_width}), instead it is  {observation_batch.size()}"
        assert action_batch.size()[1] == self._action_size, \
                f"Each action should have size {self._action_size}, not {action_batch.size()[1]}. Tensor has size {action_batch.size()}"

        #Compute 'static' features encoding
        state_s_0_batch = self.encode_static(observation_batch) #Gives a (batch_size, static_encoding_size) output

        #Compute 'dynamic' features encoding
        state_d_0_batch = self.encode_dynamic(observation_batch) #Gives a (batch_size, dyn_encoding_size) output

        state_d_1_batch, reward_d_1_batch = self.predict_dynamics(state_d_0_batch, action_batch)
        #state_d_1_batch now has size (batch_size, dyn_encoding_size)
        #reward_d_1_batch now has size (batch_size, 1) (still 2-dimensional)

        #Will now use 'static' features vectors and predicted states to predict the observation
        observation_1_batch = self.decode(state_s_0_batch,state_d_1_batch) #Gives a (batch_size, observations_channels_num, observation_height, observation_width) output

        return observation_1_batch, reward_d_1_batch


    def encode_dynamic(self, observation_batch : th.Tensor):
        assert observation_batch.size() == (observation_batch.size()[0], self._observation_channels_num, self._observation_height, self._observation_width), \
               f"Observation size should be (Any, {self._observation_channels_num}, {self._observation_height}, {self._observation_width}), instead it is  {observation_batch.size()}"
        return self._dynEncoder(observation_batch)

    def encode_static(self, observation_batch : th.Tensor):
        assert observation_batch.size() == (observation_batch.size()[0], self._observation_channels_num, self._observation_height, self._observation_width), \
               f"Observation size should be (Any, {self._observation_channels_num}, {self._observation_height}, {self._observation_width}), instead it is  {observation_batch.size()}"
        if self._staticEncoder is not None:
            return self._staticEncoder(observation_batch)
        else:
            return th.empty([observation_batch.size()[0],0]).to(observation_batch.device)

    def decode(self, static_encoding_batch : th.Tensor, dynamic_encoding_batch : th.Tensor):
        assert static_encoding_batch.size()[0] == dynamic_encoding_batch.size()[0],  \
               f"static encoding batch and dynamic encoding batch have different sizes, respectively {static_encoding_batch.size()[0]} and {dynamic_encoding_batch.size()[0]}"
        assert dynamic_encoding_batch.size() == (dynamic_encoding_batch.size()[0], self._dyn_encoding_size), \
               f"dynamic_encoding have wrong size, should be {(dynamic_encoding_batch.size()[0], self._dyn_encoding_size)}, but it's {dynamic_encoding_batch.size()}"
        assert static_encoding_batch.size() == (static_encoding_batch.size()[0], self._static_encoding_size), \
               f"static_encoding_batch have wrong size, should be {(static_encoding_batch.size()[0], self._static_encoding_size)}, but it's {static_encoding_batch.size()}"
        
        #Combine the two vectors
        state_batch = th.cat((static_encoding_batch, dynamic_encoding_batch), 1) #Gives a (batch_size, dyn_encoding_size+static_encoding_size) output
        #Predict the observation
        return self._decoder(state_batch) #Gives a (batch_size, observations_channels_num, observation_height, observation_width) output
        


    def predict_dynamics(self, state_batch : th.Tensor, action_batch : th.Tensor):
        assert state_batch.size()[0] == action_batch.size()[0], \
               f"state batch and action batch have different sizes, respectively {state_batch.size()[0]} and {action_batch.size()[0]}"
        assert state_batch.size()[1] == self._dyn_encoding_size, \
               f"States have wrong size, should be {self._dyn_encoding_size}, but it's {state_batch.size()[1]}"
        assert action_batch.size()[1] == self._action_size, \
               f"Actions have wrong size, should be {self._action_size} but it's {action_batch.size()[1]}"

        #Concatenate states and actions
        state_action_batch = th.cat((state_batch, action_batch), 1) #Gives a (batch_size, dyn_encoding_size+action_size) output
        nextstate_reward_batch = self._dynamics_net(state_action_batch) #Gives a (batch_size, dyn_encoding_size+1) output
        nextstate_batch, reward_batch = th.split(nextstate_reward_batch, [self._dyn_encoding_size, 1], 1)
        #nextstate_batch now has size (batch_size, dyn_encoding_size)
        #reward_batch now has size (batch_size, 1) (still 2-dimensional)
        return nextstate_batch, reward_batch
    



    def preprocess_observations(self, observation_batch : th.Tensor):
        resized_batch = self._resizeToInput(observation_batch)
        # Input should be in the [0,1] range, as this is what torchvision.transforms.ToTensor does
        # We move it to [-1,1]
        normalized = resized_batch*2 - 1
        return normalized
        #return resized_batch

    def postprocess_observations(self, observation_batch : th.Tensor):
        return (observation_batch + 1)/2
