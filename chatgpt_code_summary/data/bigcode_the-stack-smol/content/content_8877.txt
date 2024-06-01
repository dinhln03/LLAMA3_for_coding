import os
from argparse import ArgumentParser
from time import time
import yaml
import numpy as np
from fx_replicator import (
    build_model, load_wave, save_wave, sliding_window, LossFunc
)
import nnabla as nn
#import nnabla_ext.cudnn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save
import tqdm


def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)

    input_timesteps = config["input_timesteps"]
    output_timesteps = config["output_timesteps"]
    batch_size = config["batch_size"]

    data = load_wave(args.input_file)
    print("data.shape is:", data.shape)
    print("data.len is:", len(data))

    """
    from nnabla.ext_utils import get_extension_context
    cuda_device_id = 0
    ctx = get_extension_context('cudnn', device_id=cuda_device_id)
    print("Context: {}".format(ctx))
    nn.set_default_context(ctx)  # Set CUDA as a default context.
    """

    # padding and rounded up to the batch multiple
    block_size = output_timesteps * batch_size
    prepad = input_timesteps - output_timesteps
    postpad = len(data) % block_size
    print("postpad", block_size - postpad)
    padded = np.concatenate((
        np.zeros(prepad, np.float32),
        data,
        np.zeros(block_size - postpad, np.float32)))
    x = sliding_window(padded, input_timesteps, output_timesteps)
    x = x[:, :, np.newaxis]
    y = np.zeros_like(x)

    batchlen = x.shape[0]
    print("x.length is:",batchlen) 

    xx = nn.Variable((batch_size ,  input_timesteps, 1))

    nn.load_parameters("best_result.h5")

    print("xx.shape is:", xx.shape)

    yy = build_model(xx)
    
    print("yy.shape is:", yy.shape)

    print("x.shape in the loop is:", x[32:32 + batch_size , : , : ].shape)

    start1 = time()

    for step in range(0, batchlen , batch_size):

        xx.d = x[step:step + batch_size , : , : ]

        yy.forward()

        y[step:step + batch_size , : , : ] = yy.d

        proc_time = time() - start1
        print(proc_time)
        print(step)
        

    y = y[:, -output_timesteps:, :].reshape(-1)[:len(data)]
    save_wave(y, args.output_file)

    print("finished\n")
    proc_time = time() - start1
    print(proc_time)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    parser.add_argument(
        "--input_file", "-i",
        help="input wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--output_file", "-o", default="./predicted.wav",
        help="output wave file (48kHz/mono, *.wav)")
    parser.add_argument(
        "--model_file", "-m",
        help="input model file (*.h5)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
