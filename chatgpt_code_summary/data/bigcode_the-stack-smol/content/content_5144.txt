#!/usr/bin/env python3

import torch
import torch.optim as optim
import os, sys
import warnings
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.fast_rl.common.utils import EarlyStopping
from common.environments import get_data
from codes.f_utils import common_utils
from common.environments import TimeUnit, TradeEnvironmentType, Action
from common.environments import UpbitEnvironment
from common.environments import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector, RandomTradeDQNActionSelector

from common.fast_rl import rl_agent, value_based_model, actions, experience_single, replay_buffer
from common.fast_rl.common import utils
from common.fast_rl.common import statistics
from rl_main.trade_main import visualizer
from common.slack import PushSlack

pusher = PushSlack()

##### NOTE #####
from codes.a_config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


def evaluate(env, agent, verbose=True):
    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(env, agent, gamma=params.GAMMA, n_step=params.N_STEP)

    done = False
    state = env.reset()
    agent_state = agent.initial_agent_state()

    episode_reward = 0.0
    num_buys = 0
    info = None
    step_idx = 0
    while not done:
        step_idx += 1
        states_input = []
        processed_state = experience_source.get_processed_state(state)
        states_input.append(processed_state)

        agent_states_input = []
        agent_states_input.append(agent_state)

        new_actions, new_agent_states = agent(states_input, agent_states_input)

        agent_state = new_agent_states[0]
        action = new_actions[0]

        if action == Action.MARKET_BUY.value:
            num_buys += 1
            if num_buys > 10:
                action_str = "BUY({0})".format(10)
            else:
                action_str = "BUY({0})".format(num_buys)
        else:
            action_str = env.get_action_meanings()[action]

        msg = "[{0:2}|{1}] OHLCV: {2}, {3}, {4}, {5}, {6:<10.1f}, Action: {7:7} --> ".format(
            step_idx,
            env.data.iloc[env.transaction_state_idx]['datetime_krw'],
            env.data.iloc[env.transaction_state_idx]['open'],
            env.data.iloc[env.transaction_state_idx]['high'],
            env.data.iloc[env.transaction_state_idx]['low'],
            env.data.iloc[env.transaction_state_idx]['final'],
            env.data.iloc[env.transaction_state_idx]['volume'],
            action_str
        )

        next_state, reward, done, info = env.step(action)

        if action in [Action.HOLD.value]:
            msg += "Reward: {0:.3f}, hold coin: {1:.1f}".format(
                reward, info["hold_coin"]
            )
        elif action == Action.MARKET_BUY.value:
            if num_buys <= 10:
                coin_krw_str = "{0:.1f}".format(info['coin_krw'])
                commission_fee_str = "{0:.1f}".format(info['commission_fee'])
            else:
                coin_krw_str = "-"
                commission_fee_str = "-"

            msg += "Reward: {0:.3f}, slippage: {1:.1f}, coin_unit_price: {2:.1f}, " \
                   "coin_krw: {3}, commission: {4}, hold coin: {5:.1f}".format(
                reward, info["slippage"], info["coin_unit_price"],
                coin_krw_str, commission_fee_str, info["hold_coin"]
            )
        elif action == Action.MARKET_SELL.value:
            msg += "Reward: {0:.3f}, slippage: {1:.1f}, coin_unit_price: {2:.1f}, " \
                   "coin_krw: {3:.1f}, commission: {4:.1f}, sold coin: {5:.1f}, profit: {6:.1f}".format(
                reward, info["slippage"], info["coin_unit_price"],
                info['coin_krw'], info['commission_fee'], info["sold_coin"], info["profit"]
            )
        else:
            raise ValueError()
        if verbose:
            print(msg)

        episode_reward += reward
        state = next_state

    if verbose:
        print("SAMPLED TRANSACTION DONE! - START DATETIME: {0}, EPISODE REWARD: {1:>8.3f}, "
              "PROFIT: {2:>10.1f}, STEPS: {3}".format(
            env.transaction_start_datetime, episode_reward, info["profit"], step_idx
        ))

    return info["profit"], step_idx


def train(coin_name, time_unit, train_env, evaluate_env):
    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    net = value_based_model.DuelingDQNSmallCNN(
        observation_shape=train_env.observation_space.shape,
        n_actions=train_env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(train_env.get_action_meanings()))

    tgt_net = value_based_model.DuelingDQNSmallCNN(
        observation_shape=train_env.observation_space.shape,
        n_actions=train_env.action_space.n
    ).to(device)

    action_selector = EpsilonGreedyTradeDQNActionSelector(epsilon=params.EPSILON_INIT, env=train_env)
    agent = rl_agent.DQNAgent(dqn_model=net, action_selector=action_selector, device=device)

    argmax_action_selector = ArgmaxTradeActionSelector(env=evaluate_env)
    evaluate_agent = rl_agent.DQNAgent(dqn_model=net, action_selector=argmax_action_selector, device=device)

    random_action_selector = RandomTradeDQNActionSelector(env=evaluate_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(
        train_env, agent, gamma=params.GAMMA, n_step=params.N_STEP
    )
    buffer = replay_buffer.ExperienceReplayBuffer(experience_source, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    step_idx = 0

    last_loss = 0.0

    evaluate_steps = []
    evaluate_dqn_total_profits = []
    evaluate_random_total_profits = []

    early_stopping = EarlyStopping(
        patience=params.STOP_PATIENCE_COUNT,
        evaluation_min_threshold=params.TRAIN_STOP_EPISODE_REWARD,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_file_prefix=params.ENVIRONMENT_ID.value + "_" + coin_name + "_" + time_unit.value,
        agent=agent
    )

    with utils.SpeedTracker(params=params, frame=False, early_stopping=None) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            step_idx += params.TRAIN_STEP_FREQ
            last_entry = buffer.populate(params.TRAIN_STEP_FREQ)

            if epsilon_tracker:
                epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            solved = False
            if episode_rewards:
                for episode_reward in episode_rewards:
                    reward_tracker.set_episode_reward(
                        episode_reward, step_idx, action_selector.epsilon, last_info=last_entry.info,
                        last_loss=last_loss, model=net
                    )

                    if reward_tracker.done_episodes % params.TEST_PERIOD_EPISODE == 0:
                        print("#" * 200)
                        print("[TEST START]")
                        evaluate(evaluate_env, evaluate_agent)

                        evaluate_steps.append(step_idx)

                        dqn_total_profit, _ = evaluate_random(
                            "DQN", evaluate_env, evaluate_agent, num_episodes=100
                        )
                        evaluate_dqn_total_profits.append(dqn_total_profit)

                        random_total_profit, _ = evaluate_random(
                            "RANDOM", evaluate_env, random_agent, num_episodes=100
                        )
                        evaluate_random_total_profits.append(random_total_profit)

                        solved = early_stopping(dqn_total_profit, step_idx=step_idx)

                        visualizer.draw_performance(
                            evaluate_steps,
                            evaluate_dqn_total_profits,
                            evaluate_random_total_profits
                        )

                        print("[TEST END]")
                        print("#" * 200)

                    if solved:
                        break

            if solved:
                break

            optimizer.zero_grad()
            batch = buffer.sample(params.BATCH_SIZE)
            loss_v = value_based_model.calc_loss_double_dqn(batch, net, tgt_net, gamma=params.GAMMA, device=device)
            loss_v.backward()
            optimizer.step()

            draw_loss = min(1.0, loss_v.detach().item())
            last_loss = loss_v.detach().item()

            if step_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
                tgt_net.sync(net)

    return net


def evaluate_random(agent_type, env, agent, num_episodes, verbose=True):
    num_positive = 0
    num_negative = 0
    total_profit = 0.0
    total_steps = 0

    for _ in range(num_episodes):
        profit, step = evaluate(env, agent, verbose=False)
        if profit > 0:
            num_positive += 1
        else:
            num_negative += 1
        total_profit += profit
        total_steps += step

    avg_num_steps_per_episode = total_steps / num_episodes

    if verbose:
        print("###[{0:6}] POSITIVE: {1}/{3}, NEGATIVE: {2}/{3}, TOTAL PROFIT: {4:.1f}, AVG. STEP FOR EPISODE: {5:.1f}".format(
            agent_type, num_positive, num_negative, num_episodes, total_profit, avg_num_steps_per_episode
        ))

    return total_profit, avg_num_steps_per_episode


def evaluate_sequential_all(agent_type, env, agent, data_size, verbose=True):
    num_positive = 0
    num_negative = 0
    total_profit = 0.0
    total_steps = 0

    num_episodes = 0
    env.transaction_state_idx = 0
    while True:
        num_episodes += 1
        profit, step = evaluate(env, agent, verbose=False)
        if profit > 0:
            num_positive += 1
        else:
            num_negative += 1
        total_profit += profit
        total_steps += step

        if env.transaction_state_idx >= data_size - 1:
            break

    avg_num_steps_per_episode = total_steps / num_episodes

    if verbose:
        print("###[{0:6}] POSITIVE: {1}/{3}, NEGATIVE: {2}/{3}, TOTAL PROFIT: {4:.1f}, AVG. STEP FOR EPISODE: {5:.1f}".format(
            agent_type, num_positive, num_negative, num_episodes, total_profit, avg_num_steps_per_episode
        ))

    return num_positive, num_negative, num_episodes, total_profit, avg_num_steps_per_episode


def main():
    coin_name = "OMG"
    time_unit = TimeUnit.ONE_HOUR

    train_data_info, evaluate_data_info = get_data(coin_name=coin_name, time_unit=time_unit)

    print(train_data_info["first_datetime_krw"], train_data_info["last_datetime_krw"])
    print(evaluate_data_info["first_datetime_krw"], evaluate_data_info["last_datetime_krw"])

    train_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=train_data_info,
        environment_type=TradeEnvironmentType.TRAIN
    )

    evaluate_random_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=evaluate_data_info,
        environment_type=TradeEnvironmentType.TEST_RANDOM,
    )

    net = train(coin_name, time_unit, train_env, evaluate_random_env)

    print("#### TEST SEQUENTIALLY")
    evaluate_sequential_env = UpbitEnvironment(
        coin_name=coin_name,
        time_unit=time_unit,
        data_info=evaluate_data_info,
        environment_type=TradeEnvironmentType.TEST_SEQUENTIAL,
    )

    argmax_action_selector = ArgmaxTradeActionSelector(env=evaluate_sequential_env)
    evaluate_agent = rl_agent.DQNAgent(dqn_model=net, action_selector=argmax_action_selector, device=device)
    sequential_dqn_num_positives = []
    sequential_dqn_num_negatives = []
    sequential_dqn_num_episodes = []
    sequential_dqn_num_steps_per_episode = []
    sequential_dqn_total_profits = []
    for _ in range(10):
        num_positive, num_negative, num_episodes, total_profit, avg_num_steps_per_episode = evaluate_sequential_all(
            "DQN", evaluate_sequential_env, evaluate_agent, data_size=len(evaluate_data_info["data"]), verbose=False
        )
        sequential_dqn_num_positives.append(num_positive)
        sequential_dqn_num_negatives.append(num_negative)
        sequential_dqn_num_episodes.append(num_episodes)
        sequential_dqn_total_profits.append(total_profit)
        sequential_dqn_num_steps_per_episode.append(avg_num_steps_per_episode)

    dqn_msg = f"SEQUENTIAL: DQN - {np.mean(sequential_dqn_num_episodes):.1f} EPISODES - " \
              f"POSITIVE: {np.mean(sequential_dqn_num_positives):.1f}, " \
              f"NEGATIVE: {np.mean(sequential_dqn_num_negatives):.1f}, " \
              f"AVERAGE PROFIT {np.mean(sequential_dqn_total_profits):.1f}/STD {np.std(sequential_dqn_total_profits):.1f}, " \
              f"AVERAGE STEP {np.mean(sequential_dqn_num_steps_per_episode):.1f}"
    print(dqn_msg)

    random_action_selector = RandomTradeDQNActionSelector(env=evaluate_sequential_env)
    random_agent = rl_agent.DQNAgent(dqn_model=None, action_selector=random_action_selector, device=device)
    sequential_random_num_positives = []
    sequential_random_num_negatives = []
    sequential_random_num_episodes = []
    sequential_random_num_steps_per_episode = []
    sequential_random_total_profits = []
    for _ in range(10):
        num_positive, num_negative, num_episodes, total_profit, avg_num_steps_per_episode = evaluate_sequential_all(
            "RANDOM", evaluate_sequential_env, random_agent, data_size=len(evaluate_data_info["data"]), verbose=False
        )
        sequential_random_num_positives.append(num_positive)
        sequential_random_num_negatives.append(num_negative)
        sequential_random_num_episodes.append(num_episodes)
        sequential_random_total_profits.append(total_profit)
        sequential_random_num_steps_per_episode.append(avg_num_steps_per_episode)

    random_msg = f"SEQUENTIAL: RANDOM - {np.mean(sequential_random_num_episodes):.1f} EPISODES - " \
                 f"POSITIVE: {np.mean(sequential_random_num_positives):.1f}, " \
                 f"NEGATIVE: {np.mean(sequential_random_num_negatives):.1f}, " \
                 f"AVERAGE PROFIT {np.mean(sequential_random_total_profits):.1f}/STD {np.std(sequential_random_total_profits):.1f}, " \
                 f"AVERAGE STEP {np.mean(sequential_random_num_steps_per_episode):.1f}"
    print(random_msg)

    pusher.send_message(
        "me", dqn_msg
    )

    pusher.send_message(
        "me", random_msg
    )

if __name__ == "__main__":
    main()