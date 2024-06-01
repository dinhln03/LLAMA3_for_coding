import copy

import gym
import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictReplayBuffer
from railrl.launchers.launcher_util import run_experiment
# from railrl.samplers.data_collector import MdpPathCollector
# from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.samplers.data_collector.path_collector import ObsDictPathCollector
from railrl.samplers.data_collector.step_collector import ObsDictStepCollector
from railrl.visualization.video import VideoSaveFunctionBullet
from railrl.misc.buffer_save import BufferSaveFunction

from railrl.torch.networks import (
    CNN,
    MlpQfWithObsProcessor,
    Split,
    FlattenEach,
    Concat,
    Flatten,
)
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)

import os.path as osp
from experiments.avi.env_wrappers import FlatEnv

PARENT_DIR = '/media/avi/data/Work/github/'
import sys
env_file = osp.join(PARENT_DIR, 'avisingh599/google-research/dql_grasping/')
sys.path.insert(1, env_file)
from grasping_env import KukaGraspingProceduralEnv


def experiment(variant):

    env_params = dict(
        block_random=0.3,
        camera_random=0,
        simple_observations=False,
        continuous=True,
        remove_height_hack=True,
        render_mode="DIRECT",
        # render_mode="GUI",
        num_objects=5,
        max_num_training_models=900,
        target=False,
        test=False,
    )
    expl_env = FlatEnv(KukaGraspingProceduralEnv(**env_params))
    eval_env = expl_env
    img_width, img_height = eval_env.image_shape
    num_channels = 3

    action_dim = int(np.prod(eval_env.action_space.shape))
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=img_width,
        input_height=img_height,
        input_channels=num_channels,
        added_fc_input_size=0,
        output_conv_channels=True,
        output_size=None,
    )

    qf_cnn = CNN(**cnn_params)
    qf_obs_processor = nn.Sequential(
        qf_cnn,
        Flatten(),
    )

    qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    qf_kwargs['obs_processor'] = qf_obs_processor
    qf_kwargs['output_size'] = 1
    qf_kwargs['input_size'] = (
            action_dim + qf_cnn.conv_output_flat_size
    )
    qf1 = MlpQfWithObsProcessor(**qf_kwargs)
    qf2 = MlpQfWithObsProcessor(**qf_kwargs)

    target_qf_cnn = CNN(**cnn_params)
    target_qf_obs_processor = nn.Sequential(
        target_qf_cnn,
        Flatten(),
    )

    target_qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
    target_qf_kwargs['obs_processor'] = target_qf_obs_processor
    target_qf_kwargs['output_size'] = 1
    target_qf_kwargs['input_size'] = (
            action_dim + target_qf_cnn.conv_output_flat_size
    )

    target_qf1 = MlpQfWithObsProcessor(**target_qf_kwargs)
    target_qf2 = MlpQfWithObsProcessor(**target_qf_kwargs)

    action_dim = int(np.prod(eval_env.action_space.shape))
    policy_cnn = CNN(**cnn_params)
    policy_obs_processor = nn.Sequential(
        policy_cnn,
        Flatten(),
    )
    policy = TanhGaussianPolicyAdapter(
        policy_obs_processor,
        policy_cnn.conv_output_flat_size,
        action_dim,
        **variant['policy_kwargs']
    )

    observation_key = 'image'
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        **variant['eval_path_collector_kwargs']
    )
    replay_buffer = ObsDictReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        observation_key=observation_key,
    )

    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'batch':
        expl_path_collector = ObsDictPathCollector(
            expl_env,
            policy,
            observation_key=observation_key,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    elif variant['collection_mode'] == 'online':
        expl_path_collector = ObsDictStepCollector(
            expl_env,
            policy,
            observation_key=observation_key,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    else:
        raise NotImplementedError

    video_func = VideoSaveFunctionBullet(variant)
    algorithm.post_train_funcs.append(video_func)

    # dump_buffer_func = BufferSaveFunction(variant)
    # algorithm.post_train_funcs.append(dump_buffer_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        trainer_kwargs=dict(
            discount=0.99,
            # soft_target_tau=5e-3,
            # target_update_period=1,
            soft_target_tau=1.0,
            target_update_period=1000,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=15,
            num_epochs=5000,
            num_eval_steps_per_epoch=45,
            num_expl_steps_per_train_loop=300,
            num_trains_per_train_loop=300,
            min_num_steps_before_training=10*300,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[32, 32],
            paddings=[1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
        ),
        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        dump_video_kwargs=dict(
            imsize=48,
            save_video_period=1,
        ),
        logger_config=dict(
            snapshot_gap=10,
        ),
        dump_buffer_kwargs=dict(
            dump_buffer_period=50,
        ),
        replay_buffer_size=int(5E5),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
        use_robot_state=False,
        randomize_env=True,
    )

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env", type=str, required=True,
    #                     choices=('SawyerReach-v0', 'SawyerGraspOne-v0'))
    # parser.add_argument("--obs", required=True, type=str, choices=('pixels', 'pixels_debug'))
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    variant['env'] = 'KukaGraspingProceduralEnv'
    variant['obs'] = 'pixels'

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    exp_prefix = 'railrl-bullet-{}-{}'.format(variant['env'], variant['obs'])

    # n_seeds = 5
    # mode = 'ec2'
    # exp_prefix = 'railrl-bullet-sawyer-image-reach'

    search_space = {
        'shared_qf_conv': [
            True,
            # False,
        ],
        'collection_mode': [
            # 'batch',
            'online',
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                gpu_id=args.gpu,
                unpack_variant=False,
            )
