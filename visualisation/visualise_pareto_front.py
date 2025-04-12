import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from agent.pcn.pcn import choose_action
from scenario.create_fair_env import *
from scenario.pcn_model import *


MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/experiments/results/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/distance_metric_none/seed_0/6obj_3days_crashed/model_9.pt"  # your single best model


def plot_pareto_fronts_sbs():
    csv_files = ["sbs.csv", "sbs1.csv", "sbs2.csv", "sbs3.csv", "sbs4.csv"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))

    df_fixed = pd.read_csv("fixed.csv")
    axs[0].scatter(df_fixed["o_0"], df_fixed["o_1"], s=5, alpha=0.7, label="fixed", marker='o')

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df["o_2"].min())
        axs[0].scatter(df["o_0"], df["o_1"], s=5, alpha=0.7, label=csv_file, marker='o')
        axs[1].scatter(df["o_0"], df["o_2"], s=5, alpha=0.7, label=csv_file, marker='o')

    axs[0].set_ylim(-2000, 0)
    axs[0].set_xlabel("Hospitalizations")
    axs[0].set_ylabel("Social burden")
    axs[0].set_title("Pareto Front: Hospitalizations vs Social Burden")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Hospitalizations")
    axs[1].set_ylabel("Social burden score")
    axs[1].set_title("Pareto Front: Hospitalizations vs Social Burden Score")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("pareto_front_sbs.png")

def plot_pareto_fronts_abfta():
    csv_files = ["abfta.csv", "abfta1.csv", "abfta2.csv", "abfta3.csv", "abfta4.csv"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))

    df_fixed = pd.read_csv("fixed.csv")
    axs[0].scatter(df_fixed["o_0"], df_fixed["o_1"], s=5, alpha=0.7, label="fixed", marker='o')

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df["o_2"].min())
        axs[0].scatter(df["o_0"], df["o_1"], s=5, alpha=0.7, label=csv_file, marker='o')
        axs[1].scatter(df["o_0"], df["o_2"], s=5, alpha=0.7, label=csv_file, marker='o')

    axs[0].set_ylim(-2000, 0)
    axs[0].set_xlabel("Hospitalizations")
    axs[0].set_ylabel("Social burden")
    axs[0].set_title("Pareto Front: Hospitalizations vs Social Burden")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Hospitalizations")
    axs[1].set_ylabel("Age-based fairness (ABFTA)")
    axs[1].set_title("Pareto Front: Hospitalizations vs Age-Based Fairness")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("pareto_front_abfta.png")


device = "cpu"


def run_episode(env, model, desired_return, desired_horizon, max_return, gamma=1.0):
    """
    Runs one episode in the environment using the model's actions.
    Returns a list of Transition objects (obs, action, reward, next_obs, done).
    """
    transitions = []
    obs = env.reset()
    done = False

    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, eval=True)
        next_obs, reward, done, _ = env.step(action)

        transitions.append(
            Transition(
                observation=obs[0],
                action=env.action(action),
                reward=np.float32(reward).copy(),
                next_observation=next_obs[0],
                terminal=done
            )
        )
        obs = next_obs

        # Decrement the desired return by actual reward, clipping so it does not exceed max_return
        desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
        # Keep desired horizon from going below 1
        desired_horizon = np.float32(max(desired_horizon - 1, 1.0))

    # Optionally, discount the rewards backward
    for i in reversed(range(len(transitions) - 1)):
        transitions[i].reward += gamma * transitions[i + 1].reward

    return transitions


def plot_episode(transitions):
    """
    Basic plotting function that mimics eval_pcn's style, focusing on typical Covid env states.
    Adjust indices or subplots based on your environment's observation layout.
    """
    states = np.array([t.observation for t in transitions])
    # Add the last next_observation
    states = np.concatenate([states, transitions[-1].next_observation[None]], axis=0)

    # Example indexes for hospital, ICU, deaths, etc.
    # Adjust to match how your environment organizes states
    i_hosp_new = states[..., -3].sum(axis=-1)
    i_icu_new = states[..., -2].sum(axis=-1)
    d_new = states[..., -1].sum(axis=-1)

    actions = np.array([t.action for t in transitions])
    actions = np.concatenate([actions, [actions[-1]]], axis=0)  # Just so the final step plots an action

    # Three vertical plots
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

    # Hospital, ICU
    axs[0].plot(i_hosp_new, label="hospital")
    axs[0].plot(i_icu_new, label="icu")
    axs[0].legend()
    axs[0].set_ylabel("Hosp / ICU")

    # Deaths
    axs[1].plot(d_new, label="deaths", color="red")
    axs[1].legend()
    axs[1].set_ylabel("Deaths")

    # Actions
    axs[2].plot(actions[:, 0], label="action_0")
    if actions.shape[1] > 1:
        axs[2].plot(actions[:, 1], label="action_1")
    if actions.shape[1] > 2:
        axs[2].plot(actions[:, 2], label="action_2")
    axs[2].legend()
    axs[2].set_ylabel("Actions")
    axs[2].set_xlabel("Timesteps")

    plt.tight_layout()
    plt.show()


def evaluate_model():
    """
    Evaluate an existing PCN model without parsing CLI arguments.
    Customize any default arguments here as needed.
    """

    # ------------------------------------------------------------------
    # 2) Hardcode defaults for environment creation
    #    (Mirrors typical usage from create_fair_covid_env)
    # ------------------------------------------------------------------
    class SimpleArgs:
        # For the Covid environment
        env = "covid"
        seed = 0
        model = "densebig"  # used by create_fair_covid_env to pick the net shape
        action = "continuous"  # continuous or discrete
        budget = None  # if you want to enforce a budget, set an integer
        episode_length = 100
        bias = 0
        ignore_sensitive = False
        # ... add other fields from create_fair_env if needed ...

    args = SimpleArgs()

    # Specify which reward indices we want to keep (like in your usage):
    # E.g. if the environment uses 5D or 6D rewards, pick a subset
    # For example, if you want [0,1,2,3,4]
    rewards_to_keep = [0, 1, 2, 3, 4]

    # Create env
    env, scale, ref_point, scaling_factor, max_return_array, ss, se, sa, nA, with_budget = create_fair_covid_env(
        args, rewards_to_keep
    )
    print("Environment created:", env)

    # ------------------------------------------------------------------
    # 3) Load your existing, trained PCN model
    #    Adjust the path to your .pt file as needed
    # ------------------------------------------------------------------
    loaded_model_path = "path/to/my_trained_model.pt"
    model = torch.load(loaded_model_path, map_location=device)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Evaluate the model
    #    We'll define some default desired_return and horizon
    #    You can pick any target returns or run multiple episodes
    # ------------------------------------------------------------------
    desired_return = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # shape matches your reward dimension
    desired_horizon = 17  # e.g. if you want to plan for 30 steps
    max_return = np.array(max_return_array, dtype=np.float32)

    # Number of episodes
    n_episodes = 2
    for i in range(n_episodes):
        transitions = run_episode(env, model, desired_return.copy(), desired_horizon, max_return, gamma=1.0)
        print(f"\nEpisode {i + 1}: final discounted returns = {transitions[0].reward}")
        # Plot results from the episode
        plot_episode(transitions)



if __name__ == "__main__":
    #plot_pareto_fronts_sbs()
    #plot_pareto_fronts_abfta()
    evaluate_model()
