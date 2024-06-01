import copy

from functools import wraps

import numpy as np

import wandb
import torchvision
import torch
import torch.nn.functional as F

from kornia import enhance, filters
from torchvision.transforms import RandomApply, RandomChoice
from atariari.methods.utils import EarlyStopping

from torch import nn
from torch.utils.data import BatchSampler, RandomSampler


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils


# class RandomApply(nn.Module):
#     def __init__(self, fn, p):
#         super().__init__()
#         self.fn = fn
#         self.p = p

#     def forward(self, x):
#         if random.random() > self.p:
#             return x
#         return self.fn(x)


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer  # final avg-pooling layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection

# main class


class BYOL(nn.Module):
    def __init__(self, net, image_size, grayscale=True, num_frame_stack=1, batch_size=64, hidden_layer=-2, projection_size=256, projection_hidden_size=4096, augment_fn=None, augment_fn2=None, moving_average_decay=0.99, wandb=None, patience=15):
        super().__init__()

        # default SimCLR augmentation

        #####
        # IMPORTANT for kornia: parameters are often float!! e.g. 1. vs 1
        # DEFAULT_AUG = nn.Sequential(
        # RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        # augs.RandomHorizontalFlip(),
        # RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        # input tensor: float + normalized range [0,1]
        # augs.RandomResizedCrop(
        #     size=(image_size, image_size), scale=(0.84, 1.), ratio=(1.,1.), p=1.0)
        # augs.Normalize(mean=torch.tensor(
        #     [0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        # )

        kernel_size = (9, 9)  # has to be ODD
        kernel_std = np.random.uniform(low=0.1, high=2.0)
        kernel_std = (kernel_std,)*2
        aug_transform = torchvision.transforms.Compose([
            RandomChoice(
                [enhance.AdjustBrightness(0.4),
                 enhance.AdjustBrightness(0.3),
                 enhance.AdjustBrightness(0.2),
                 enhance.AdjustBrightness(0.1),
                 enhance.AdjustBrightness(0.0)]
            ),
            RandomChoice(
                [enhance.AdjustContrast(1.0),
                 enhance.AdjustContrast(0.9),
                 enhance.AdjustContrast(0.8),
                 enhance.AdjustContrast(0.7),
                 enhance.AdjustContrast(0.6)]
            ),
            RandomApply([filters.GaussianBlur2d(
                kernel_size, kernel_std)], p=0.5)
            # RandomChoice(
            #     [enhance.AdjustContrast(1.0),
            #      enhance.AdjustContrast(1.0),
            #      enhance.AdjustContrast(1.0),
            #      filters.GaussianBlur2d((1, 1), (1, 1)),
            #      filters.GaussianBlur2d((3, 3), (1.5, 1.5))]
            # )
        ])

        self.augment1 = default(augment_fn, aug_transform)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size)

        self.batch_size = batch_size
        # get device of network and make wrapper same device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device is {self.device.type}")
        self.to(self.device)
        self.wandb = wandb
        self.early_stopper = EarlyStopping(
            patience=patience, verbose=False, wandb=self.wandb, name="encoder-byol")

        if self.wandb:
            wandb.watch(self.online_encoder, self.target_encoder,
                        self.online_predictor)
        # send a mock image tensor to instantiate singleton parameters
        assert grayscale
        nr_channels = num_frame_stack
        self.forward(torch.rand(batch_size, nr_channels,
                                210, 160, device=self.device))
        self.opt = torch.optim.Adam(self.parameters(), lr=3e-4)
        print(
            f"Finished Initialization of BYOL with model {self.online_encoder.net.__class__.__name__}")

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder, self.online_encoder)

    def forward(self, x):
        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

    def logResults(self, epoch_idx, epoch_loss, prefix=""):
        print(f"{prefix} Epoch: {epoch_idx}, Loss: {epoch_loss}")
        if self.wandb:
            self.wandb.log({prefix + '_loss': epoch_loss},
                           step=epoch_idx, commit=False)

    def doOneEpoch(self, nr_epoch, episodes):
        mode = "train" if self.training else "val"
        data_generator = generate_batch(episodes, self.batch_size, self.device)
        for steps, batch in enumerate(data_generator):
            print(f"batch nr {steps} for mode {mode}")
            loss = self(batch)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.update_moving_average()  # update moving average of target encoder
        self.logResults(nr_epoch, loss / steps, prefix=mode)
        if mode == "val":
            self.early_stopper(-loss / steps, self.online_encoder)


def generate_batch(episodes, batch_size, device):
    total_steps = sum([len(e) for e in episodes])
    print('Total Steps: {}'.format(total_steps))
    # Episode sampler
    # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
    sampler = BatchSampler(RandomSampler(range(len(episodes)),
                                         replacement=True, num_samples=total_steps),
                           batch_size, drop_last=True)
    for nr, indices in enumerate(sampler):
        x = []
        episodes_batch = [episodes[i] for i in indices]
        # print(f"indices in sampler nr {nr} are {*indices,}")
        for e in episodes_batch:
            t = np.random.randint(0, len(e))
            x.append(e[t])
        yield torch.stack(x).float().to(device) / 255.  # SCALING!!!!