import argparse
import datetime
import json
import os
import time
from os import path

import numpy as np
import torch
from absl import flags
from torch import optim
from pprint import pprint
import wandb

from src.alive_sieve import AliveSieve, SievePlayback
from src.nets import AgentModel
from src.rewards_lib import calc_rewards
from src.sampling import (generate_test_batches,
                          generate_training_batch,
                          hash_batches)

FLAGS = flags.FLAGS


def render_action(t, s, prop, term):
    agent = t % 2
    speaker = 'A' if agent == 0 else 'B'
    utility = s.utilities[:, agent]
    print('  ', end='')
    if speaker == 'B':
        print('                                   ', end='')

    print(' ' + ''.join([str(v) for v in s.m_prev[0].view(-1).tolist()]), end='')
    print(' %s/%s %s/%s %s/%s' % (
        prop[0][0].item(), s.pool[0][0].item(),
        prop[0][1].item(), s.pool[0][1].item(),
        prop[0][2].item(), s.pool[0][2].item(),
    ), end='')
    print('')

    if t + 1 == s.N[0]:
        print(' [out of time]')
    elif term[0][0]:
        print(' ACC')


def save_model(model_file, agent_models, agent_opts, start_time, episode):
    state = {}
    for i in range(2):
        state['agent%s' % i] = {}
        state['agent%s' % i]['model_state'] = agent_models[i].state_dict()
        state['agent%s' % i]['opt_state'] = agent_opts[i].state_dict()
    state['episode'] = episode
    state['elapsed_time'] = time.time() - start_time
    with open(model_file + '.tmp', 'wb') as f:
        torch.save(state, f)
    os.rename(model_file + '.tmp', model_file)


def load_model(model_file, agent_models, agent_opts):
    with open(model_file, 'rb') as f:
        state = torch.load(f)
    for i in range(2):
        agent_models[i].load_state_dict(state['agent%s' % i]['model_state'])
        agent_opts[i].load_state_dict(state['agent%s' % i]['opt_state'])
    episode = state['episode']
    # create a kind of 'virtual' start_time
    start_time = time.time() - state['elapsed_time']
    return episode, start_time


class State(object):
    def __init__(self, N, pool, utilities):
        batch_size = N.size()[0]
        self.N = N
        self.pool = pool
        self.utilities = torch.zeros(batch_size, 2, 3, dtype=torch.int64, device=FLAGS.device)
        self.utilities[:, 0] = utilities[0]
        self.utilities[:, 1] = utilities[1]

        self.last_proposal = torch.zeros(batch_size, 3, dtype=torch.int64, device=FLAGS.device)
        self.m_prev = torch.zeros(batch_size, FLAGS.utt_max_length, dtype=torch.int64, device=FLAGS.device)

    def sieve_(self, still_alive_idxes):
        self.N = self.N[still_alive_idxes]
        self.pool = self.pool[still_alive_idxes]
        self.utilities = self.utilities[still_alive_idxes]
        self.last_proposal = self.last_proposal[still_alive_idxes]
        self.m_prev = self.m_prev[still_alive_idxes]


def run_episode(
    batch,
    agent_models,
    batch_size,
    testing,
    render=False,
    initial_agent=0):
    """
    turning testing on means, we disable stochasticity: always pick the argmax
    """

    s = State(**batch)

    sieve = AliveSieve(batch_size=batch_size)
    actions_by_timestep = []
    alive_masks = []

    # next two tensors wont be sieved, they will stay same size throughout
    # entire batch, we will update them using sieve.out_idxes[...]
    rewards = torch.zeros(batch_size, 3, device=FLAGS.device)
    num_steps = torch.full((batch_size,), FLAGS.max_timesteps, dtype=torch.int64, device=FLAGS.device)
    term_matches_argmax_count = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    num_policy_runs = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0
    utt_mask = torch.zeros(2, batch_size, 3, dtype=torch.int64, device=FLAGS.device)
    prop_mask = torch.zeros(2, batch_size, 3, dtype=torch.int64, device=FLAGS.device)

    entropy_loss_by_agent = [
        torch.zeros(1, device=FLAGS.device),
        torch.zeros(1, device=FLAGS.device)
    ]
    if render:
        print('  ')
        print('           ',
              '{}   {}   {}'.format(*s.utilities[0][0].tolist()),
              '      ',
              '{} {} {}'.format(*s.pool[0].tolist()),
              '          ',
              '{}   {}   {}'.format(*s.utilities[0][1].tolist()))

    current_A_proposal = torch.zeros(sieve.batch_size, 3, dtype=torch.int64, device=FLAGS.device)
    prev_A_proposal = torch.zeros(sieve.batch_size, 3, dtype=torch.int64, device=FLAGS.device)
    current_A_message = torch.zeros(sieve.batch_size, FLAGS.utt_max_length, dtype=torch.int64, device=FLAGS.device)
    prev_A_message = torch.zeros(sieve.batch_size, FLAGS.utt_max_length, dtype=torch.int64, device=FLAGS.device)
    current_A_term = torch.zeros(sieve.batch_size, 1, dtype=torch.uint8)

    for t in range(FLAGS.max_timesteps):
        if FLAGS.linguistic:
            if FLAGS.normal_form and t % 2 == 1:
                _prev_message = prev_A_message
            else:
                _prev_message = s.m_prev
        else:
            _prev_message = torch.zeros(sieve.batch_size, 6, dtype=torch.int64, device=FLAGS.device)

        if FLAGS.proposal:
            if FLAGS.normal_form and t % 2 == 1:
                _prev_proposal = prev_A_proposal
            else:
                _prev_proposal = s.last_proposal
        else:
            _prev_proposal = torch.zeros(sieve.batch_size, 3, dtype=torch.int64, device=FLAGS.device)

        # agent = t % 2
        agent = (initial_agent + t) % 2
        agent_model = agent_models[agent]
        (nodes, term_a, s.m_prev, this_proposal, _entropy_loss,
         _term_matches_argmax_count, _utt_matches_argmax_count, _utt_stochastic_draws,
         _prop_matches_argmax_count, _prop_stochastic_draws, _utt_mask, _prop_mask) = agent_model(
             pool=s.pool,
             utility=s.utilities[:, agent],
             m_prev=_prev_message,
             prev_proposal=_prev_proposal,
             testing=testing,
         )

        entropy_loss_by_agent[agent] += _entropy_loss
        actions_by_timestep.append(nodes)
        term_matches_argmax_count += _term_matches_argmax_count
        num_policy_runs += sieve.batch_size
        utt_matches_argmax_count += _utt_matches_argmax_count
        utt_stochastic_draws += _utt_stochastic_draws
        prop_matches_argmax_count += _prop_matches_argmax_count
        prop_stochastic_draws += _prop_stochastic_draws

        if FLAGS.force_masking_comm:
            utt_mask[agent][sieve.out_idxes] |= _utt_mask
            prop_mask[agent][sieve.out_idxes] |= _prop_mask

        if FLAGS.proposal_termination and not FLAGS.normal_form:
            term_a = torch.prod(this_proposal == _prev_proposal,
                                dim=1,
                                keepdim=True)
        elif not FLAGS.proposal_termination and FLAGS.normal_form:
            #TODO which proposal to use here?
            if t % 2 == 1:
                term_a = (term_a * current_A_term)
            else:
                current_A_term = term_a
                term_a = torch.zeros((sieve.batch_size,1), dtype=torch.uint8, device=FLAGS.device)

        elif FLAGS.proposal_termination and FLAGS.normal_form:
            if t % 2 == 1:
                term_a = torch.prod(this_proposal == current_A_proposal,
                                    dim=1,
                                    keepdim=True)
            else:
                term_a = torch.zeros((sieve.batch_size,1), dtype=torch.uint8, device=FLAGS.device)


        if render and sieve.out_idxes[0] == 0:
            render_action(
                t=t,
                s=s,
                term=term_a,
                prop=this_proposal
            )

        new_rewards = calc_rewards(
            t=t,
            s=s,
            term=term_a,
            agent=agent,
        )
        rewards[sieve.out_idxes] = new_rewards
        s.last_proposal = this_proposal

        if FLAGS.normal_form and t % 2 == 0:
            prev_A_proposal = current_A_proposal
            current_A_proposal = this_proposal
            prev_A_message = current_A_message
            current_A_message = s.m_prev


        sieve.mark_dead(term_a)
        sieve.mark_dead(t + 1 >= s.N)
        alive_masks.append(sieve.alive_mask.clone())
        sieve.set_dead_global(num_steps, t + 1)
        if sieve.all_dead():
            break

        s.sieve_(sieve.alive_idxes)

        if FLAGS.normal_form:
            current_A_proposal = current_A_proposal[sieve.alive_idxes]
            prev_A_proposal = prev_A_proposal[sieve.alive_idxes]
            current_A_message = current_A_message[sieve.alive_idxes]
            prev_A_message = prev_A_message[sieve.alive_idxes]


        sieve.self_sieve_()

    if render:
        print(' rewards: {:2.2f} {:2.2f} {:2.2f}'.format(*rewards[0].tolist()))
        print('  ')

    utt_mask_count = utt_mask.sum(dim=[1,2]).cpu().numpy()
    prop_mask_count = prop_mask.sum(dim=[1,2]).cpu().numpy()

    return (actions_by_timestep, rewards, num_steps, alive_masks, entropy_loss_by_agent,
            term_matches_argmax_count, num_policy_runs, utt_matches_argmax_count, utt_stochastic_draws,
            prop_matches_argmax_count, prop_stochastic_draws, utt_mask_count, prop_mask_count)


def safe_div(a, b):
    """
    returns a / b, unless b is zero, in which case returns 0
    this is primarily for usage in cases where b might be systemtically zero, eg because comms are disabled or similar
    also accounts for a or b being tensors
    """
    if isinstance(a, torch.Tensor):
        a = a.item()
    if isinstance(b, torch.Tensor):
        b = b.item()
    return 0 if b == 0 else a / b


def run(args):
    """
    testing option will:
    - use argmax, ie disable stochastic draws
    - not run optimizers
    - not save model
    """
    if args.wandb:
        if args.wandb_offline:
            os.environ["WANDB_MODE"] = "dryrun"

        wandb.init(project='ecn',
                   name=args.name,
                   dir=f'{args.savedir}',
                   group=args.wandb_group)
        wandb.config.update(args)
        wandb.config.update(FLAGS)
    flags_dict = {flag.name: flag.value for flag in FLAGS.flags_by_module_dict()['main.py']}
    args_dict = args.__dict__
    pprint(args_dict)
    pprint(flags_dict)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_r = np.random.RandomState(args.seed)
    else:
        train_r = np.random

    test_r = np.random.RandomState(args.test_seed)
    test_batches = generate_test_batches(batch_size=args.batch_size,
                                         num_batches=5,
                                         random_state=test_r)
    test_hashes = hash_batches(test_batches)

    episode = 0
    start_time = time.time()
    agent_models = []
    agent_opts = []
    agent_name = ['A', 'B']
    for i in range(2):
        model = AgentModel(
            name=agent_name[i],
            term_entropy_reg=args.term_entropy_reg,
            utterance_entropy_reg=args.utterance_entropy_reg,
            proposal_entropy_reg=args.proposal_entropy_reg
        ).to(FLAGS.device)
        agent_models.append(model)
        agent_opts.append(optim.Adam(params=agent_models[i].parameters()))
    if args.wandb:
        wandb.watch(agent_models)
    if path.isfile(args.model_file) and not args.no_load:
        episode, start_time = load_model(
            model_file=args.model_file,
            agent_models=agent_models,
            agent_opts=agent_opts)
        print('loaded model')
    elif args.testing:
        print('')
        print('ERROR: must have loadable model to use --testing option')
        print('')
        return
    last_print = time.time()
    rewards_sum = torch.zeros(3, device=FLAGS.device)
    steps_sum = 0
    count_sum = 0
    f_log = open(args.log_file, 'w')
    all_args = {**args_dict, **flags_dict}
    f_log.write('meta: %s\n' % json.dumps(all_args))
    last_save = time.time()
    baseline = torch.zeros(3, device=FLAGS.device)
    term_matches_argmax_count = 0
    num_policy_runs = 0
    utt_matches_argmax_count = 0
    utt_stochastic_draws = 0
    prop_matches_argmax_count = 0
    prop_stochastic_draws = 0
    utt_mask_count = np.array([0,0])
    prop_mask_count = np.array([0,0])
    while episode < args.episodes:
        render = (episode % args.render_every_episode == 0)
        split = 2 if FLAGS.randomize_first else 1
        agent_losses = [0,0]
        both_rewards = []

        for i in range(2):
            agent_opts[i].zero_grad()

        for initial_agent in range(split):
            batch = generate_training_batch(batch_size=args.batch_size // split,
                                            test_hashes=test_hashes,
                                            random_state=train_r)
            (actions, rewards, steps, alive_masks, entropy_loss_by_agent,
             _term_matches_argmax_count, _num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws,
             _prop_matches_argmax_count, _prop_stochastic_draws,
             _utt_mask_count, _prop_mask_count) = run_episode(
                 batch=batch,
                 agent_models=agent_models,
                 batch_size=args.batch_size // split,
                 render=render,
                 initial_agent=initial_agent,
                 testing=args.testing)
            term_matches_argmax_count += _term_matches_argmax_count
            utt_matches_argmax_count += _utt_matches_argmax_count
            utt_stochastic_draws += _utt_stochastic_draws
            num_policy_runs += _num_policy_runs
            prop_matches_argmax_count += _prop_matches_argmax_count
            prop_stochastic_draws += _prop_stochastic_draws
            utt_mask_count += _utt_mask_count
            prop_mask_count += _prop_mask_count

            if not args.testing:
                reward_loss_by_agent = [0, 0]
                baselined_rewards = rewards - baseline
                rewards_by_agent = []
                for i in range(2):
                    if FLAGS.prosocial:
                        rewards_by_agent.append(baselined_rewards[:, 2])
                    else:
                        rewards_by_agent.append(baselined_rewards[:, i])
                sieve_playback = SievePlayback(alive_masks)
                for t, global_idxes in sieve_playback:
                    agent = (initial_agent + t) % 2
                    if len(actions[t]) > 0:
                        for action in actions[t]:
                            _rewards = rewards_by_agent[agent]
                            _reward = _rewards[global_idxes].float().contiguous().view(
                                sieve_playback.batch_size, 1)
                            _reward_loss = - (action * _reward)
                            _reward_loss = _reward_loss.sum()
                            reward_loss_by_agent[agent] += _reward_loss

                for i in range(2):
                    loss = entropy_loss_by_agent[i] + reward_loss_by_agent[i]
                    loss.backward()

            rewards_sum += rewards.detach().sum(0)
            steps_sum += steps.sum()
            count_sum += args.batch_size // split
            both_rewards.append(rewards)


        for i in range(2):
            agent_opts[i].step()

        rewards = torch.cat(both_rewards).detach()
        baseline = 0.7 * baseline + 0.3 * rewards.mean(0).detach()

        if render:
            """
            run the test batches, print the results
            """
            test_rewards_sum = np.zeros(3)
            test_count_sum = len(test_batches) * args.batch_size
            test_num_policy_runs = 0
            test_utt_mask_count = [0,0]
            test_prop_mask_count = [0,0]
            test_utt_mask_count = np.array([0,0])
            test_prop_mask_count = np.array([0,0])
            for test_batch in test_batches:
                (actions, test_rewards, steps, alive_masks, entropy_loss_by_agent,
                 _term_matches_argmax_count, _test_num_policy_runs, _utt_matches_argmax_count, _utt_stochastic_draws,
                 _prop_matches_argmax_count, _prop_stochastic_draws,
                 _test_utt_mask_count, _test_prop_mask_count) = run_episode(
                     batch=test_batch,
                     agent_models=agent_models,
                     batch_size=args.batch_size,
                     render=True,
                     testing=True)
                test_rewards_sum += test_rewards.sum(0).cpu().numpy()
                test_num_policy_runs += _test_num_policy_runs
                test_utt_mask_count += _test_utt_mask_count
                test_prop_mask_count += _test_prop_mask_count

            time_since_last = time.time() - last_print
            rewards_str = '%.2f,%.2f,%.2f' % (rewards_sum[0] / count_sum,
                                              rewards_sum[1] / count_sum,
                                              rewards_sum[2] / count_sum)
            test_rewards_str = '%.2f,%.2f,%.2f' % (test_rewards_sum[0] / test_count_sum,
                                                   test_rewards_sum[1] / test_count_sum,
                                                   test_rewards_sum[2] / test_count_sum)
            baseline_str = '%.2f,%.2f,%.2f' % (baseline[0], baseline[1], baseline[2])
            utt_mask_pct = utt_mask_count / (3 * count_sum)
            test_utt_mask_pct = test_utt_mask_count / (3 * test_count_sum)
            prop_mask_pct = prop_mask_count / (3 * count_sum)
            test_prop_mask_pct = test_prop_mask_count / (3 * test_count_sum)
            print('test  {}'.format(test_rewards_str))
            print('train {}'.format(rewards_str))
            print('base  {}'.format(baseline_str))
            print('ep {}, {} games/sec, {:2.2f} avg steps'.format(
                episode,
                int(count_sum / time_since_last),
                steps_sum.item() / count_sum
            ))
            print('argmaxp term={:4.4f} utt={:4.4f} prop={:4.4f}'.format(
                term_matches_argmax_count / num_policy_runs,
                safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                prop_matches_argmax_count / prop_stochastic_draws
            ))
            if FLAGS.force_masking_comm:
                print('utt mask % {:2.2f},{:2.2f} test % {:2.2f},{:2.2f}'.format(
                    *utt_mask_pct, *test_utt_mask_pct,
                ))
                print('prop mask % {:2.2f},{:2.2f} test % {:2.2f},{:2.2f}'.format(
                    *prop_mask_pct, *test_prop_mask_pct,
                ))

            episode_log = {
                'episode': episode,
                'avg_reward_A': (rewards_sum[0] / count_sum).item(),
                'avg_reward_B': (rewards_sum[1] / count_sum).item(),
                'avg_reward_0': (rewards_sum[2] / count_sum).item(),
                'test_reward_A': (test_rewards_sum[0] / test_count_sum).item(),
                'test_reward_B': (test_rewards_sum[1] / test_count_sum).item(),
                'test_reward': (test_rewards_sum[2] / test_count_sum).item(),
                'avg_steps': torch.true_divide(steps_sum, count_sum).item(),
                'games_sec': (count_sum / time_since_last),
                'elapsed': time.time() - start_time,
                'argmaxp_term': term_matches_argmax_count / num_policy_runs,
                'argmaxp_utt': safe_div(utt_matches_argmax_count, utt_stochastic_draws),
                'argmaxp_prop': prop_matches_argmax_count / prop_stochastic_draws,
                'utt_unmasked_A': utt_mask_pct[0],
                'utt_unmasked_B': utt_mask_pct[1],
                'prop_unmasked_A': prop_mask_pct[0],
                'prop_unmasked_B': prop_mask_pct[1],
                'test_utt_unmasked_A': test_utt_mask_pct[0],
                'test_utt_unmasked_B': test_utt_mask_pct[1],
                'test_prop_unmasked_A': test_prop_mask_pct[0],
                'test_prop_unmasked_B': test_prop_mask_pct[1],
            }
            f_log.write(json.dumps(episode_log) + '\n')
            f_log.flush()
            if args.wandb:
                wandb.log(episode_log)
            last_print = time.time()
            steps_sum = 0
            rewards_sum.fill_(0)
            term_matches_argmax_count = 0
            num_policy_runs = 0
            utt_matches_argmax_count = 0
            utt_stochastic_draws = 0
            prop_matches_argmax_count = 0
            prop_stochastic_draws = 0
            count_sum = 0
            utt_mask_count.fill(0)
            prop_mask_count.fill(0)

        if (not args.testing
            and not args.no_save
            and episode > 0
            and episode % args.save_every_episode == 0):
            save_model(model_file=args.model_file,
                       agent_models=agent_models,
                       agent_opts=agent_opts,
                       start_time=start_time,
                       episode=episode)
            print('saved model')

        episode += 1

    if (not args.no_save and
        not args.testing):
        save_model(
            model_file=args.model_file,
            agent_models=agent_models,
            agent_opts=agent_opts,
            start_time=start_time,
            episode=episode)
        print('saved model')
    f_log.close()
