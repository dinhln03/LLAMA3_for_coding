import os
import pickle

import numpy as np
from tqdm import tqdm

from deeptutor.envs.DashEnv import *
from deeptutor.envs.EFCEnv import EFCEnv
from deeptutor.envs.HRLEnv import *
from deeptutor.infrastructure.utils import *
from deeptutor.tutors.LeitnerTutor import LeitnerTutor
from deeptutor.tutors.RandTutor import RandTutor
from deeptutor.tutors.PPOTutor import PPOTutor
from deeptutor.tutors.SACTutor import SACTutor
from deeptutor.tutors.DQNTutor import DQNTutor
from deeptutor.tutors.MLPTRPOTutor import MLPTRPOTutor
from deeptutor.tutors.GRUTRPOTutor import GRUTRPOTutor
from deeptutor.tutors.SuperMnemoTutor import SuperMnemoTutor
from deeptutor.tutors.ThresholdTutor import ThresholdTutor


def load_rewards(tutor_name, data_dir):
    filename = os.path.join(data_dir, f"{tutor_name}_reward_logs.pkl")
    if not os.path.exists(filename):
        return {}
    with open(filename, "rb") as f:
        return pickle.load(f)["rewards"]


def main():
    override = True # override existing data
    data_dir = os.path.join(os.getcwd(), "data")
    n_steps = 200
    n_items = 30
    const_delay = 5
    discount = 0.99
    n_reps = 10
    n_eps = 100
    env_kwargs = {
        "n_items": n_items,
        "n_steps": n_steps,
        "discount": discount,
        "sample_delay": sample_const_delay(const_delay),
    }
    reward_funcs = [
        "likelihood",
        "log_likelihood"
    ]
    envs = [
        ("EFC", EFCEnv),
        ("HLR", HLREnv),
        ("DASH", DASHEnv)
    ]

    tutor_builders = [
        # ("Random", RandTutor),
        # ("Leitner", LeitnerTutor),
        # ("SuperMnemo", SuperMnemoTutor),
        # ("Threshold", ThresholdTutor),
        # ("MLPTRPO", MLPTRPOTutor),
        # ("GRUTRPO", GRUTRPOTutor),
        # ("PPO", PPOTutor),
        ("DQN", DQNTutor),
    ]

    rl_tutors = [MLPTRPOTutor, GRUTRPOTutor, PPOTutor, DQNTutor]

    reward_logs = {
        "n_steps": n_steps,
        "n_items": n_items,
        "discount": discount,
        "const_delay": const_delay,
        "n_reps": n_reps,
        "n_eps": n_eps,
        "reward_funcs": reward_funcs,
    }

    for i, (tutor_name, build_tutor) in enumerate(tutor_builders):
        print(f"Training {tutor_name}")
        rewards = load_rewards(tutor_name, data_dir)
        for h, (base_env_name, base_env) in enumerate(envs):
            for m, reward_func in enumerate(reward_funcs):
                env_name = (
                    base_env_name + "-" + ("L" if reward_func == "likelihood" else "LL")
                )
                print(f"Environment: {env_name}")
                if env_name in rewards.keys() and not override:
                    print("Skipping\n")
                    continue
                R = np.zeros((n_eps, n_reps))
                for j in tqdm(range(n_reps)):
                    np.random.seed(j)
                    env = base_env(**env_kwargs, reward_func=reward_func)
                    if build_tutor in rl_tutors:
                        rl_env = make_rl_student_env(env)
                        agent = build_tutor(n_items)
                        R[:, j] = agent.train(rl_env, n_eps=n_eps, seed=j)
                    else:
                        if "Thresh" in tutor_name:
                            agent = build_tutor(n_items, env=env)
                        else:
                            agent = build_tutor(n_items)
                        R[:, j] = agent.train(env, n_eps=n_eps)
                rewards[env_name] = R
                reward_logs["rewards"] = rewards
                with open(os.path.join(data_dir, f"{tutor_name}_reward_logs.pkl"), "wb") as f:
                    pickle.dump(reward_logs, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
