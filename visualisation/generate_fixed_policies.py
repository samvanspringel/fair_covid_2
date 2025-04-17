import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym_covid import *
from visualise_pareto_front import compute_sbs, compute_abfta
from pathlib import Path
from agent.pcn.pcn import choose_action, non_dominated
from scenario.create_fair_env import *
from scenario.pcn_model import *
from fairness.individual.individual_fairness import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import h5py

def plot_coverage_set(files, measure):
    plt.figure(figsize=(8, 6))  # Create one figure

    for file in files:
        if "h5" in file:
            df_fixed = get_df_pareto_front_log()
        else:
            df_fixed = pd.read_csv(file)

        # Check if mean of first column > 3000, and scale if needed
        if 'o_0' in df_fixed.columns and 'o_1' in df_fixed.columns:
            x = df_fixed['o_0'].values
            y = df_fixed['o_1'].values
        else:
            x = df_fixed.iloc[:, 0].values
            y = df_fixed.iloc[:, 1].values

        plt.scatter(x, y,
                    label=f"{file}", marker='o')

    plt.xlabel("Hospitalizations (possibly scaled by 1e4)")
    plt.ylabel(f"Measure {measure}")
    plt.title(f"Coverage Set Fixed Policies for Measure {measure}")
    plt.legend()
    plt.ylim(-1500, 0)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def interpolate_run(run, w=100):
    # Create a common x_grid from all runs’ first column values
    # Create a uniform x grid
    all_steps = np.linspace(run[:, 0].min(), run[:, 0].max())

    # Interpolate y-values over that grid
    all_values = np.interp(all_steps, run[:, 0], run[:, 1])

    return all_steps, all_values

def get_df_pareto_front_log():
    logdir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/visualisation/seed_9_budget_0"
    logdir_path = Path(logdir)
    for path in logdir_path.rglob('log.h5'):
        with h5py.File(path, 'r') as logfile:
            pareto_front = logfile['train/leaves/ndarray'][-1]
            _, pareto_front_i = non_dominated(pareto_front[:, [0, 1]], return_indexes=True)
            pf = pareto_front[pareto_front_i]

            #pf = pf[pf[:, 1] >= extreme_y_threshold]
            pf = pf[np.argsort(pf[:, 0])]  # sort by first dimension (objective 0)

            x, y = interpolate_run(pf)

            pf = np.vstack([x, y]).T * np.array([10000, 90])

            df = pd.DataFrame(pf, columns=['o_0', 'o_1'])

    return df

def generate_fixed_coverage_set(env, fairness, amount_of_policies=100):
    """
    Runs a series of episodes under fixed policies and returns an array with:
      [u, cumulative_hospitalizations, cumulative_lost_contacts]

    Parameters:
      env: an instance of your EpiEnv environment.
      n_fixed: number of fixed policies to evaluate. (Each policy corresponds to a fixed u in [0,1])
      n_steps: number of timesteps to simulate for the episode.

    Returns:
      A NumPy array of shape (n_fixed, 3)
    """
    policy_results = []
    # Create n_fixed fixed policies: equally spaced between 0 and 1.
    us = np.linspace(0, 1, amount_of_policies)
    for u in us:
        fixed_action = np.array([u, u, u], dtype=np.float32)
        #print(f"Executing fixed policy {fixed_action}")
        # Reset the environment to the initial state.
        env.reset()
        done = False
        fairness_states = []
        # Initialize accumulators for the rewards.
        cumulative_reward = np.array([0.0, 0.0])
        # Run simulation for n_steps timesteps.
        while not done:
            obs, reward, done, info = env.step(fixed_action)
            if fairness == "SBS" or fairness == "ABFTA":
                fairness_states.append(env.state_df())
            cumulative_reward += reward
        # In your env, r_arh is computed as negative hospitalizations,
        # so flip the sign to get a positive number.
        hospitalizations = cumulative_reward[0]
        if fairness == "SBS":
            y_value = compute_sbs(fairness_states)
        elif fairness == "ABFTA":
            y_value = compute_abfta(fairness_states, distance_metric="kl")
        else:
            y_value = cumulative_reward[1]
        policy_results.append([hospitalizations, y_value])

    return np.array(policy_results)


if __name__ == '__main__':
    y_measure = "ABFTA"
    # env = gym.make(f'BECovidWithLockdownODEContinuous-v0')
    # coverage_set = generate_fixed_coverage_set(env, y_measure, amount_of_policies=100)
    #
    # # Create a DataFrame and save as a CSV file:
    # df = pd.DataFrame(coverage_set, columns=["hospitalizations", "measure"])
    # df.to_csv(f"fixed_{y_measure}.csv", index=False)
    # print("Saved fixed policies")
    #plot_coverage_set([f"fixed_{y_measure}.csv"], y_measure)
    #plot_coverage_set(["test.csv", "cs_fixed.csv"], y_measure)
    plot_coverage_set(["log.h5", "fixed_sb.csv"], "SB")
