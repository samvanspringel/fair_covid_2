from itertools import combinations

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym_covid import *
from visualise_pareto_front import compute_sbs, compute_abfta, get_scaling_plot
from pathlib import Path
from agent.pcn.pcn import choose_action, non_dominated
from scenario.create_fair_env import *
from scenario.pcn_model import *
from fairness.individual.individual_fairness import *
from collections import deque

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

            # pf = pf[pf[:, 1] >= extreme_y_threshold]
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
        print(f"Executing fixed policy {fixed_action}")
        # Reset the environment to the initial state.
        env.reset()
        done = False
        # Initialize accumulators for the rewards.
        cumulative_reward = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Run simulation for n_steps timesteps.
        while not done:
            obs, reward, done, info = env.step(fixed_action)

            sbs = compute_sbs([env.state_df()])

            abfta = compute_abfta([env.state_df()])

            reward = np.append(reward, [sbs, abfta])

            cumulative_reward += reward
        # In your env, r_arh is computed as negative hospitalizations,
        # so flip the sign to get a positive number.
        hospitalizations = cumulative_reward[1]
        if fairness == "sbs":
            y_value = cumulative_reward[6]
        elif fairness == "abfta":
            y_value = cumulative_reward[7]
        else:
            y_value = cumulative_reward[5]
        policy_results.append([hospitalizations, y_value])

    return np.array(policy_results)


def generate_fixed_coverage_set_new(env, fairness, amount_of_policies=100):
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
        print(f"Executing fixed policy {fixed_action}")
        # Reset the environment to the initial state.
        env.reset()
        done = False
        cumulative_reward = np.array([0.0, 0.0])
        # Sliding window for state-diff pairs
        window_size = 17
        fairness_states = deque(maxlen=window_size)
        total_diff = 0.0
        while not done:
            obs, reward, done, info = env.step(fixed_action)
            cumulative_reward += reward
            if fairness == "sbs":
                # Add current state & diff pair to window
                fairness_states.append(env.state_df())
                # Compute current SBS over the window and accumulate
                diff = compute_sbs(fairness_states)
                total_diff += diff
            elif fairness == "abfta":
                # existing ABFTA logic (unchanged)
                pass
        hospitalizations = cumulative_reward[0]
        if fairness == "sbs":
            y_value = total_diff
        elif fairness == "abfta":
            y_value = compute_abfta(fairness_states, distance_metric="kl")
        else:
            y_value = cumulative_reward[1]
        policy_results.append([hospitalizations, y_value])

    return np.array(policy_results)


def test_scenarios(hospitalization_risks):
    """Run a handful of hand-crafted reduction-vs-risk scenarios to see metric behavior."""
    import numpy as np

    idx_min = np.argmin(hospitalization_risks)
    idx_max = np.argmax(hospitalization_risks)
    # Precompute risk-probabilities
    risk = hospitalization_risks
    risk_prob = risk / (np.sum(risk) + 1e-12)
    risk_prob = [0.08, 0.02, 0.05, 0.04, 0.07, 0.09, 0.12, 0.21, 0.13, 0.21]

    print("\n=== Running scenario tests ===")
    scenarios = {}

    # 1) High-risk group gets LOW reduction; everyone else = 1
    arr = [0.0, 0.35, 0.20, 0.25, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]
    scenarios["high risk LOW reduction: unfair"] = arr

    # 2) High-risk group gets HIGH reduction; everyone else = 0
    arr = [0.0, 0.4, 0.0, 0.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.0]
    scenarios["UNFAIR"] = arr

    # 3) Low-risk group gets LOW reduction; everyone else = 1
    arr = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.3, 0.2, 0.1, 0.3]
    scenarios["low risk LOW reduction: fair"] = arr

    # 4) Low-risk group gets HIGH reduction; everyone else = 0
    arr = [0.0, 0.5, 0.20, 0.2, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]
    scenarios["low risk HIGH reduction: unfair"] = arr

    # 5) Perfect proportional: reductions ∝ risk
    scenarios["perfect proportional: fair"] = risk.copy()

    # 6) Inverse proportional: higher-risk gets lower reduction
    arr = [0.09, 0.21, 0.13, 0.21, 0.12, 0.08, 0.07, 0.8, 0.05, 0.02]
    scenarios["inverse proportional: fair?"] = arr

    for name, red in scenarios.items():
        red = np.array(red, dtype=float)
        red_prob = red / (np.sum(red) + 1e-12)

        # compute metric f
        f = 0.0
        for i in range(len(red_prob)):
            for j in range(len(red_prob)):
                if i != j:
                    D = abs(red_prob[j] - red_prob[i])
                    d = abs(risk_prob[j] - risk_prob[i])
                    f += abs(d - D)

        print(f"\nScenario: {name}")
        print("  Reduction distribution (sum=1):", red_prob)
        print("  Risk distribution     (sum=1):", risk_prob)
        print(f"  Metric f = {f:.6f}")
    print("=== End scenario tests ===\n")

def env_random_demo():
    import gym
    from fairness.individual.individual_fairness import get_reduction_impact

    env = gym.make("BECovidWithLockdownODEBudget5Continuous-v0")

    np.set_printoptions(precision=5, suppress=True, linewidth=400)
    n_samples = 1000000


    # run our fixed scenario tests once
    hospitalization_risks = env.model.get_hospitalization_risk()
    #test_scenarios(hospitalization_risks)
    max_f = 0
    min_f = 100000
    step = 0
    for _ in range(n_samples):
        env.reset()
        done = False

        while not done:
            step += 1
            print(f"Step: {step} --- Max: {max_f}")
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

            reduction_per_age_group = np.abs(get_reduction_impact(env.C_diff_fairness))
            hospitalization_risks = env.model.get_hospitalization_risk()

            # Convert to positive totals per group (already abs):
            red = reduction_per_age_group
            # Normalize to probability distribution (sum to 1)
            red_sum = np.sum(red)
            red_prob = red / (red_sum + 1e-12)
            # print("Reductions per age-group:", red)
            # print("Reduction distribution (sum=1):", red_prob)

            # Normalize hospitalization risks to probability distribution
            risk = hospitalization_risks
            risk_sum = np.sum(risk)
            risk_prob = risk / (risk_sum + 1e-12)
            # print("Hospitalization risks:", risk)
            # print("Risk distribution (sum=1):", risk_prob)
            # f = 0
            # for i in range(len(red_prob)):
            #     for j in range(len(risk_prob)):
            #         if i != j:
            #             D = np.abs(red_prob[j] - red_prob[i])
            #             d = np.abs(risk_prob[j] - risk_prob[i])
            #             f += np.abs(d - D)
            # print("Original: ", f)

            # optimized computation of f via vectorized differences
            D_diff = np.abs(red_prob[:, None] - red_prob[None, :])
            d_diff = np.abs(risk_prob[:, None] - risk_prob[None, :])
            mask = ~np.eye(D_diff.shape[0], dtype=bool)
            f = np.sum(np.abs(D_diff[mask] - d_diff[mask]))

            if f > max_f:
                max_f = f
            if f < min_f:
                min_f = f


    print("Max: ", max_f)
    print("Min: ", min_f)








if __name__ == "__main__":
    env_random_demo()

if __name__ == '__main__':
    y_measure = "abfta"
    env_type = "ODE"
    budget = 5
    env = gym.make(f'BECovidWithLockdown{env_type}Budget{budget}Continuous-v0')
    # coverage_set = generate_fixed_coverage_set(env, y_measure, amount_of_policies=100)

    # Create a DataFrame and save as a CSV file:
    # df = pd.DataFrame(coverage_set, columns=["hospitalizations", "measure"])
    # df.to_csv(f"fixed_policy_{y_measure}.csv", index=False)
    # print(f"Saved fixed policies in fixed_policy_{y_measure}.csv")
    # plot_coverage_set([f"fixed_policy_{y_measure}.csv"], y_measure)
    # plot_coverage_set(["testcov.csv", "fixed_sb.csv"], y_measure)
    # plot_coverage_set(["window17.csv", f"fixed_{y_measure}.csv"], "SBS")
