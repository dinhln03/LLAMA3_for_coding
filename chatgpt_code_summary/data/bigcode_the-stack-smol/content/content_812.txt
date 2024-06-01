import copy
import torch
import logging
import numpy as np
from sacred import Experiment

from noge.data_loaders import get_datasets, get_test_loader, get_train_generator
from noge.factory import make_env, make_memory
from noge.network import make_network
from noge.agent import Actor, main_loop, loop_ing
from noge.trainers import DQNTrainer, Replay
from noge.policies import LinearSchedule, GraphDQNPolicy
from noge.preprocessors import Preprocessor
from noge.evaluation import Evaluator, eval_ing
from noge.constants import CONFIGS_DIR, EVAL_DIR
from xlog.utils import get_logger
from xlog.mlflow_observer import MlflowObserver

ex = Experiment(name='NOGE_DQN', ingredients=[eval_ing, loop_ing])
ex.add_config(str(CONFIGS_DIR / 'dqn.yaml'))
ex.logger = get_logger(__name__, level=logging.INFO)
ex.observers = [MlflowObserver(tracking_uri=str(EVAL_DIR.absolute()))]


@ex.automain
def train(dataset, test_size, max_episode_steps, reward_type, input_meas_type, meas_transform,
          target_transform, node_history, gamma, target_update_freq,
          cat_features, feature_range, replay_capacity, min_horizon, epsilon_start, epsilon_end,
          exploration_frac, n_train_steps, train_freq, loss, batch_size, lr, n_test_episodes, init_eval,
          n_eval_artifacts, test_freq, log_freq, device, seed, data_seed, save_model, _log, _run, _config):
    np.set_printoptions(precision=2, suppress=True)

    if device.startswith('cuda'):
        assert torch.cuda.is_available()

    logger = _log
    device = torch.device(device)

    # data source
    train_set, test_set = get_datasets(dataset, seed=data_seed, test_size=test_size)
    max_nodes = max(train_set.max_nodes, test_set.max_nodes)
    max_edges = 2 * max(train_set.max_edges, test_set.max_edges)  # for undirected graphs, consider both directions

    test_loader = get_test_loader(test_set, seed=seed, num_samples=n_test_episodes)
    train_gen = get_train_generator(train_set, seed=seed)

    preprocessor = Preprocessor(input_meas_type=input_meas_type,
                                output_meas_type=input_meas_type,
                                feature_range=feature_range,
                                meas_transform=meas_transform,
                                target_transform=target_transform,
                                temporal_offsets=[1.],
                                max_nodes=max_nodes,
                                device=device)

    # environment
    train_env_config = dict(
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
        max_nodes=max_nodes,
        max_edges=max_edges,
        nn_feat='N' in cat_features,
    )

    train_env = make_env(**train_env_config, data_generator=train_gen, seed=seed)
    test_env_config = copy.deepcopy(train_env_config)
    test_env_config.update(sample_goals=False, data_generator=None)
    test_env = make_env(**test_env_config, seed=seed)

    # graph memory + graph preprocessing
    neg_label, pos_label = feature_range
    mem_features = dict(cat=cat_features)
    graph_mem_config = dict(
        max_episode_steps=max_episode_steps,
        max_nodes=max_nodes,
        max_edges=max_edges,
        history=node_history,
        memory_type='cat',
        features=mem_features,
        neg_label=neg_label,
        pos_label=pos_label
    )

    eval_memory = make_memory(online=True, **graph_mem_config)
    acting_memory = make_memory(online=True, **graph_mem_config)

    # model
    model_config = dict(
        dim_node=eval_memory.dim_node,
        dim_meas=preprocessor.dim_input_meas,
        dim_goal=1,
        max_edges=max_edges,
        **_config['model']
    )

    network = make_network(**model_config).to(device)

    # evaluation
    eval_policy = GraphDQNPolicy(network, eval_memory, preprocessor=preprocessor, device=device)
    evaluator = Evaluator(test_loader, test_env, eval_policy)

    # experience collecting policy
    exploration_steps = int(exploration_frac * n_train_steps)
    exploration_schedule = LinearSchedule(epsilon_start, epsilon_end, exploration_steps)
    acting_policy = GraphDQNPolicy(network,
                                   graph_memory=acting_memory,
                                   preprocessor=preprocessor,
                                   exploration_schedule=exploration_schedule,
                                   device=device)

    # replay buffer
    replay_buffer = Replay(capacity=replay_capacity,
                           ob_space=train_env.observation_space,
                           graph_mem_config=graph_mem_config,
                           min_horizon=min_horizon)

    # actor: runs the simulation forward and stores to the replay buffer
    actor = Actor(train_env, acting_policy, replay_buffer)

    # trainer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    if loss == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    trainer = DQNTrainer(gamma=gamma,
                         target_update_freq=target_update_freq,
                         replay_buffer=replay_buffer,
                         batch_size=batch_size,
                         network=network,
                         preprocessor=preprocessor,
                         criterion=criterion,
                         optimizer=optimizer,
                         device=device)

    # fill up the replay buffer
    network.eval()
    logger.info(f"Filling up the replay buffer...")
    actor.step(n=replay_capacity, use_tqdm=True)
    logger.info(f"Replay buffer filled: [{len(replay_buffer)} / {replay_capacity}]")

    # fit the preprocessor with buffer data
    preprocessor.fit(replay_buffer._measurements)

    best_perf = main_loop(actor, trainer, evaluator, network, exploration_schedule,
                          init_eval, n_eval_artifacts, n_train_steps, train_freq, log_freq, test_freq, save_model)

    train_env.close()
    evaluator.close()

    return best_perf
