from agent.pcn.pcn import non_dominated, Transition, choose_action, epsilon_metric
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd


device = 'cpu'


def run_episode(env, model, desired_return, desired_horizon, max_return):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, eval=True)
        n_obs, reward, done, info = env.step(action)
        if 'action' in info:
            action = info['action']

        transitions.append(Transition(
            observation=obs[1],
            action=env.action(action),
            reward=np.float32(reward).copy(),
            next_observation=n_obs[1],
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        # reward = np.array((reward[1], reward[2]))
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))
    return transitions


def run_fixed_episode(env, actions, desired_return, desired_horizon, max_return):
    transitions = []
    obs = env.reset()
    done = False; step = 0
    while not done:
        action = actions[step]
        n_obs, reward, done, info = env.step(action)
        if 'action' in info:
            action = info['action']

        transitions.append(Transition(
            observation=obs[0],
            action=env.action(action),
            reward=np.float32(reward).copy(),
            next_observation=n_obs[0],
            terminal=done
        ))

        obs = n_obs
        step += 1
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        # reward = np.array((reward[1], reward[2]))
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))
    return transitions


def plot_episode(transitions, alpha=1.):
    states = np.array([t.observation for t in transitions])
    # add final state
    states = np.concatenate((states, transitions[-1].next_observation[None]), axis=0)
    ari = (states[:-1,:,0]-states[1:,:,0]).sum(axis=-1)
    i_hosp_new = states[...,-3].sum(axis=-1)
    i_icu_new = states[...,-2].sum(axis=-1)
    d_new = states[...,-1].sum(axis=-1)
    actions = np.array([t.action for t in transitions])
    # append action of None
    actions = np.concatenate((actions, [[None]*3]))

    # steps in dates
    start = datetime.date(2020, 5, 3)
    week = datetime.timedelta(days=7)
    dates = [start+week*i for i in range(0, 18, 2)]
    
    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp_new, alpha=alpha, label='hosp', color='blue')
    ax.plot(i_icu_new,  alpha=alpha, label='icu', color='green')
    ax.plot(i_hosp_new+i_icu_new, label='hosp+icu',  alpha=alpha, color='orange')
    ax.plot(d_new, alpha=alpha, label='deaths', color='red')
    ax.set_xticks(ticks=np.arange(0, 18, 2), labels=[str(d.day)+'/'+str(d.month) for d in dates])

    # deaths
    ax = axs[1]
    # ax.plot(d_new, alpha=alpha, label='deaths', color='red')
    ax.plot(ari, alpha=alpha, label='ari', color='black')

    # actions
    ax = axs[2]
    ax.set_ylim([0, 1])
    ax.plot(actions[:,0], alpha=alpha, label='p_w', color='blue')
    ax.plot(actions[:,1], alpha=alpha, label='p_s', color='orange')
    ax.plot(actions[:,2], alpha=alpha, label='p_l', color='green')

    axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')
    axs[1].set_ylabel('deaths')
    axs[2].set_ylabel('actions')
    # for ax in axs:
    #     ax.legend()
    return [start+week*i for i in range(0, 18, 1)], ari, i_hosp_new, i_icu_new, d_new, actions[:, 0], actions[:, 1], actions[:, 2]


def eval_pcn(env, model, desired_return, desired_horizon, max_return, objectives, gamma=1., n=1, ode_env=None):
    plt.subplots(3, 1, sharex=True)
    alpha = 1 if n == 1 else 0.2
    returns = np.empty((n, desired_return.shape[-1]))
    all_transitions = []
    for n_i in range(n):
        transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
        # compute return
        for i in reversed(range(len(transitions)-1)):
            transitions[i].reward += gamma * transitions[i+1].reward
        
        returns[n_i] = transitions[i].reward.flatten()
        print(f'ran model with desired-return: {desired_return.flatten()}, got {transitions[i].reward.flatten()}')
        print('action sequence: ')
        for t in transitions:
            print(f'- {t.action}')
        t = plot_episode(transitions, alpha)
        t = t + tuple(zip(*[ti.reward*env.scale for ti in transitions]))

        df = pd.DataFrame([x for x in zip(*t)], columns=['dates', 'ari', 'i_hosp_new', 'i_icu_new', 'd_new', 'p_w', 'p_s', 'p_l'] + [f'o_{oi}' for oi in range(returns.shape[1])])
        # manually set p_s to 0 during school holidays
        holidays = df['dates'] >= datetime.date(2020, 7, 1)
        df.loc[holidays, 'p_s'] = 0
        all_transitions.append(df)
    # title = 'Re: '+ ' '.join([f'{o:.3f}' for o in (returns.mean(0)*env.scale)[objectives]])
    title = 'Re: '+ ' '.join([f'{o:.3f}' for o in (returns.mean(0)*env.scale)])
    title += '\n'
    title += 'Rt: '+ ' '.join([f'{o:.3f}' for o in (desired_return*env.scale)[objectives]])
    if ode_env is not None:
        actions = [np.stack((df['p_w'], df['p_s'], df['p_l']), axis=-1) for df in all_transitions]
        actions = np.mean(actions, axis=0)
        transitions_fixed = run_fixed_episode(ode_env, actions, desired_return, desired_horizon, max_return)
        t = plot_episode(transitions_fixed, 1)
        return_ = np.sum([t.reward for t in transitions_fixed], axis=0)
        title += '\nODE Re: '+ ' '.join([f'{o:.3f}' for o in (return_*env.scale)])
        
    plt.suptitle(title)
    print(f'ran model with desired-return: {desired_return[objectives].flatten()}, got average return {returns[:,objectives].mean(0).flatten()}')
    return returns, all_transitions


if __name__ == '__main__':
    import argparse
    import uuid
    import os
    import gym_covid
    import gym
    import pathlib
    import h5py

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('env', type=str, help='ode or binomial')
    parser.add_argument('model', type=str, help='load model')
    parser.add_argument('--objectives', default=[1, 5], type=int, nargs='+', help='index for ari, arh, pw, ps, pl, ptot')
    parser.add_argument('--n', type=int, default=1, help='evaluation runs')
    parser.add_argument('--interactive', action='store_true', help='interactive policy selection')
    parser.set_defaults(interactive=False)
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model)


    # import pickle
    # with open(model_dir / 'returns.pkl', 'wb') as f:
    #     all_returns = np.array(all_returns)
    #     # compute e-metric
    #     epsilon = epsilon_metric(all_returns.mean(axis=1), pareto_front)
    #     print(f'epsilon-max: {epsilon.max()}')
    #     print(f'epsilon-mean: {epsilon.mean()}')
    #     pickle.dump(all_returns, f)