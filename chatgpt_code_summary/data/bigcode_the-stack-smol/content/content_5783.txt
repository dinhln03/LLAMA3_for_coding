import imageio

import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment


def load_policy(path):
    return tf.compat.v2.saved_model.load(path)


def visualize_policy(environment, policy, output_filename, num_episodes=1, fps=5):
    rendering_environment = environment
    if isinstance(environment, tf_py_environment.TFPyEnvironment):
        # The inner env should be used for rendering
        rendering_environment = environment.pyenv.envs[0]

    with imageio.get_writer(output_filename, fps=fps) as video:
        font = ImageFont.load_default()
        total_reward = None

        def _add_environment_frame():
            rendered_env = rendering_environment.render()
            image = Image.fromarray(rendered_env.astype(np.uint8), mode='RGB')
            draw = ImageDraw.Draw(image)
            draw.text((5, 5), 'TR: %.1f' % total_reward, font=font)
            image_as_numpy = np.array(image.getdata()).reshape(rendered_env.shape).astype(np.uint8)
            video.append_data(image_as_numpy)

        for _ in range(num_episodes):
            total_reward = 0.0
            time_step = environment.reset()
            _add_environment_frame()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                total_reward += time_step.reward.numpy()[0]
                _add_environment_frame()


def evaluate_policy(env, policy, num_episodes):
    total_return = 0.0
    total_num_steps = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        episode_num_steps = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
            episode_num_steps += 1
        total_return += episode_return
        total_num_steps += episode_num_steps

    return (total_return / num_episodes).numpy()[0], total_num_steps / num_episodes


def as_tf_env(env):
    return tf_py_environment.TFPyEnvironment(env)


def create_replay_buffer(agent, train_env, replay_buffer_size):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_size,
    )


def create_collect_driver(train_env, agent, replay_buffer, collect_steps):
    return dynamic_step_driver.DynamicStepDriver(
        train_env, agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps,
    )


def cudnn_workaround():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
