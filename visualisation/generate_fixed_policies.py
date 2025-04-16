import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym_covid import *
from visualise_pareto_front import compute_sbs, compute_abfta


def plot_coverage_set(file, measure):
    # Read the CSV files
    df_fixed = pd.read_csv(file)

    plt.figure(figsize=(8, 6))
    plt.scatter(df_fixed["hospitalizations"], df_fixed["measure"],
                label=f"Fixed Policies Measure {measure}", marker='o', color='blue')
    plt.xlabel("Hospitalizations")
    plt.ylabel(f"Measure {measure}")
    plt.title(f"Coverage Set Fixed Policies for Measure {measure}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
            y_value = compute_abfta(fairness_states)
        else:
            y_value = cumulative_reward[1]
        policy_results.append([hospitalizations, y_value])

    return np.array(policy_results)


if __name__ == '__main__':
    y_measure = "ABFTA"
    env = gym.make(f'BECovidWithLockdownODEContinuous-v0')
    coverage_set = generate_fixed_coverage_set(env, y_measure, amount_of_policies=100)

    # Create a DataFrame and save as a CSV file:
    df = pd.DataFrame(coverage_set, columns=["hospitalizations", "measure"])
    df.to_csv(f"fixed_{y_measure}.csv", index=False)
    print("Saved fixed policies")
    plot_coverage_set(f"fixed_{y_measure}.csv", y_measure)
