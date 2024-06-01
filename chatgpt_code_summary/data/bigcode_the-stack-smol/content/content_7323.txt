#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
from .hyper_bodies import * 
from .hypernetwork_ops import *
from ..utils.hypernet_heads_defs import *
from ..component.samplers import *

class VanillaHyperNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaHyperNet, self).__init__()
        self.mixer = False
        self.config = VanillaNet_config(body.feature_dim, output_dim)
        self.fc_head = LinearGenerator(self.config['fc_head'])
        self.body = body
        self.to(Config.DEVICE)

    def sample_model_seed(self):
        if not self.mixer:
            self.model_seed = {
                    'fc_head_z': torch.rand(self.fc_head.config['n_gen'], particles, self.z_dim).to(Config.DEVICE)
            }
        else:
            self.model_seed = torch.rand(particles, self.s_dim)
    
    def forward(self, x, z=None):
        phi = self.body(tensor(x, z))
        y = self.fc_head(z[0], phi)
        return y


class DuelingHyperNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body, hidden, dist, particles):
        super(DuelingHyperNet, self).__init__()
        self.mixer = False
        
        self.config = DuelingNet_config(body.feature_dim, action_dim)
        self.config['fc_value'] = self.config['fc_value']._replace(d_hidden=hidden)
        self.config['fc_advantage'] = self.config['fc_advantage']._replace(d_hidden=hidden)
        self.fc_value = LinearGenerator(self.config['fc_value']).cuda()
        self.fc_advantage = LinearGenerator(self.config['fc_advantage']).cuda()
        self.features = body
        
        self.s_dim = self.config['s_dim']
        self.z_dim = self.config['z_dim']
        self.n_gen = self.config['n_gen']
        self.particles = particles
        self.noise_sampler = NoiseSampler(dist, self.z_dim, self.particles)
        # self.sample_model_seed()
        self.to(Config.DEVICE)
    
    def sample_model_seed(self):
        sample_z = self.noise_sampler.sample().to(Config.DEVICE)
        # sample_z = sample_z.unsqueeze(0).repeat(self.features.config['n_gen'], 1, 1)
        sample_z = sample_z.unsqueeze(0).repeat(self.particles, 1)
        self.model_seed = {
            'value_z': sample_z,
            'advantage_z': sample_z,
        }
    
    def set_model_seed(self, seed):
        self.model_seed = seed

    def forward(self, x, to_numpy=False, theta=None):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        if x.shape[0] == 1 and x.shape[1] == 1: ## dm_env returns one too many dimensions
            x = x[0]
        phi = self.body(x)
        return self.head(phi)

    def body(self, x=None):
        if not isinstance(x, torch.cuda.FloatTensor):
            x = tensor(x)
        return self.features(x)

    def head(self, phi):
        phi = phi.repeat(self.particles, 1, 1)  # since we have a deterministic body with many heads
        value = self.fc_value(self.model_seed['value_z'], phi)
        advantage = self.fc_advantage(self.model_seed['advantage_z'], phi)
        q = value.expand_as(advantage) + (advantage - advantage.mean(-1, keepdim=True).expand_as(advantage))
        return q

    def sample_model(self, component):
        param_sets = []
        if component == 'q':
            param_sets.extend(self.fc_value(z=self.model_seed['value_z']))
            param_sets.extend(self.fc_advantage(z=self.model_seed['advantage_z']))
        return param_sets

    def predict_action(self, x, pred, to_numpy=False):
        x = tensor(x)
        q = self(x)
        if pred == 'max':
            max_q, max_q_idx = q.max(-1)  # max over q values
            max_actor = max_q.max(0)[1]  # max over particles
            action = q[max_actor].argmax()
        
        elif pred == 'rand':
            idx = np.random.choice(self.particles, 1)[0]
            action = q[idx].max(0)[1]
        
        elif pred == 'mean':
            action_means = q.mean(0)  #[actions]
            action = action_means.argmax()

        if to_numpy:
            action = action.cpu().detach().numpy()
        return action


