import torch
import os
import os.path
import shutil
import numpy as np
import soundfile as sf

from pathlib import PurePath
from torch import nn
from torch.utils.data import DataLoader, random_split
from asteroid.data import TimitDataset
from asteroid.data.utils import CachedWavSet, RandomMixtureSet, FixedMixtureSet
from tqdm import tqdm

from torch import optim
from pytorch_lightning import Trainer, seed_everything, loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from asteroid_filterbanks.transforms import mag
from asteroid.engine import System
from asteroid.losses import singlesrc_neg_sisdr

from egs.whamr.TasNet.model import TasNet

BATCH_SIZE       = 8     # could be more on cluster, test if larger one work
SAMPLE_RATE      = 8000   # as agreed upon
CROP_LEN         = 24000  # average track len in TIMIT
SEED             = 42     # magic number :)

def sisdr_loss_wrapper(est_target, target):
    return singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()

def train_val_split(ds, val_fraction=0.1, random_seed=SEED):
    assert val_fraction > 0 and val_fraction < 0.5
    len_train = int(len(ds) * (1 - val_fraction))
    len_val = len(ds) - len_train
    return random_split(ds, [len_train, len_val], generator=torch.Generator().manual_seed(random_seed))

DRONE_NOISE_DIR = '/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/noises-train-drones'
# fixed SNRs for validation set
TRAIN_SNRS = [-25, -20, -15, -10, -5]


TIMIT_DIR = PurePath('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/TIMIT')
TIMIT_DIR_8kHZ = PurePath('/jmain01/home/JAD007/txk02/aaa18-txk02/Datasets/TIMIT_8kHZ')

# Reproducibility - fix all random seeds
seed_everything(SEED)

# Load noises, resample and save into the memory
noises = CachedWavSet(DRONE_NOISE_DIR, sample_rate=SAMPLE_RATE, precache=True)

# Load clean data and split it into train and val
timit = TimitDataset(TIMIT_DIR_8kHZ, subset='train', sample_rate=SAMPLE_RATE, with_path=False)
timit_train, timit_val = train_val_split(timit, val_fraction=0.1, random_seed=SEED)

# Training data mixes crops randomly on the fly with random SNR in range (effectively infinite training data)
# `repeat_factor=20` means that the dataset contains 20 copies of itself - it is the easiest way to make the epoch longer
timit_train = RandomMixtureSet(timit_train, noises, random_seed=SEED, snr_range=(-25, -5),
                               crop_length=CROP_LEN, repeat_factor=30)

# Validation data is fixed (for stability): mix every clean clip with all the noises in the folder
# Argument `mixtures_per_clean` regulates with how many different noise files each clean file will be mixed
timit_val = FixedMixtureSet(timit_val, noises, snrs=TRAIN_SNRS, random_seed=SEED,
                            mixtures_per_clean=5, crop_length=CROP_LEN)

NUM_WORKERS = 5
train_loader = DataLoader(timit_train, shuffle=True, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(timit_val, batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS, drop_last=True)

# some random parameters, does it look sensible?
LR = 1e-3
REDUCE_LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 20
MAX_EPOCHS = 20

# the model here should be constructed in the script accordingly to the passed config (including the model type)
# most of the models accept `sample_rate` parameter for encoders, which is important (default is 16000, override)
model = TasNet(fb_conf={'n_filters': 512, 'kernel_size': 40, 'stride': 20},
                     mask_conf ={'n_layers': 4, 'n_units': 500, 'dropout': 0.3, "n_src": 1})

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=REDUCE_LR_PATIENCE)
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)
checkpoint = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.2f}',
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        verbose=True
    )

# Probably we also need to subclass `System`, in order to log the target metrics on the validation set (PESQ/STOI)
system = System(model, optimizer, sisdr_loss_wrapper, train_loader, val_loader, scheduler)

# log dir and model name are also part of the config, of course
LOG_DIR = 'logs'
logger = pl_loggers.TensorBoardLogger(LOG_DIR, name='TIMIT-drones-TasNet-random_test', version=1)

# choose the proper accelerator for JADE, probably `ddp` (also, `auto_select_gpus=True` might be useful)
trainer = Trainer(max_epochs=MAX_EPOCHS, gpus=-1,
                  logger=logger, callbacks=[early_stopping, checkpoint], deterministic=True, gradient_clip_val=5.0,)

trainer.fit(system)
#torch.save(model.serialize(), 'tasnet_model.pt')
