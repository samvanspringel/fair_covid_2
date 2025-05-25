import itertools
from itertools import combinations
import os

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gym_covid import *
from pathlib import Path

from matplotlib import patches

from agent.pcn.pcn import choose_action, non_dominated
from scenario.create_fair_env import *
from scenario.pcn_model import *
from fairness.individual.individual_fairness import *
from collections import deque
from visualise_pareto_front import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import h5py


# Sampling helpers
from scipy.stats import qmc
from sklearn.cluster import KMeans

label_colors = {
    "constrasting": "plum",
    "random": "lightskyblue",
    "fixed": "palegreen"
}

phi0 = np.array([0.972, 0.992, 0.984, 0.987, 0.977, 0.971, 0.958, 0.926, 0.956, 0.926])
delta2_star = 0.756
hospitalization_risks = (1 - phi0) * delta2_star
def compute_sbs(states):
    fairness_window = 0

    for state_C_diff in states:
        state_df, C_diff = state_C_diff
        # Convert Series to NumPy arrays
        S = state_df["S"].to_numpy()
        R = state_df["R"].to_numpy()
        h = state_df["h_risk"].to_numpy()

        A = (S + R) / h

        term_matrix = A[:, None] + A[None, :]
        C_diff = np.abs(C_diff)


        # Multiply each slice of C_diff by term_matrix and sum over i and j
        fairness = np.sum(C_diff * term_matrix, axis=(1, 2))
        fairness_window += fairness.sum()
    return fairness_window * (-1)

def get_kl_distribution(states, label):
    all_kl_vals = []
    for state_df, C_diff in states:
        hospitalization_risks = state_df["h_risk"].values  # shape (K,)
        reduction_per_age_group = np.abs(get_reduction_impact(C_diff))

        # 1) Normalize reduction_per_age_group so each of the 10 rows sums to 1
        reduction_sums = reduction_per_age_group.sum(axis=1, keepdims=True)
        reduction_sums[reduction_sums == 0] = 1e-12
        reduction_distributions = reduction_per_age_group / reduction_sums

        # 2) Construct pair‑wise KL‑divergence matrix
        K = reduction_distributions.shape[0]
        KL_D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                KL_D[i, j] = kl_divergence(reduction_distributions[i], reduction_distributions[j], 0)

        # 3) Convert hospitalization_risks to a probability distribution
        risk_prob = hospitalization_risks / (hospitalization_risks.sum() + 1e-12)

        # 4) Construct the pair‑wise risk‑difference matrix
        H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))  # element (i,j) = risk_prob[i] - risk_prob[j]

        H_d_MAX = H_d.max()
        H_d_MIN = H_d.min()

        KL_D_scaled = ((KL_D - KL_D_MIN) / (KL_D_MAX - KL_D_MIN)) * (H_d_MAX - H_d_MIN) + H_d_MIN

        kl_vals = KL_D_scaled[~np.eye(K, dtype=bool)]
        all_kl_vals.extend(kl_vals)

    # Compute and plot aggregated PDF of all pairwise KL divergences
    pdf_vals, bin_edges = np.histogram(all_kl_vals, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.figure()
    plt.plot(bin_centers, pdf_vals, linewidth=1, color=label_colors[label.lower()])
    plt.xlabel(f'KL Divergence')
    plt.ylabel('Probability Density')
    plt.ylim(0, 80)
    plt.title(f'KL Divergence for {label} actions')
    plt.show()

def compute_abfta(states):
    fairness = 0.08
    KL_D_MAX = 28
    KL_D_MIN = 0

    for state_df, C_diff in states:
        hospitalization_risks = state_df["h_risk"].values  # shape (K,)
        reduction_per_age_group = np.abs(get_reduction_impact(C_diff))

        # 1) Normalize reduction_per_age_group so each of the 10 rows sums to 1
        reduction_sums = reduction_per_age_group.sum(axis=1, keepdims=True)
        reduction_sums[reduction_sums == 0] = 1e-12
        reduction_distributions = reduction_per_age_group / reduction_sums

        # 2) Construct pair‑wise KL‑divergence matrix
        K = reduction_distributions.shape[0]
        KL_D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                KL_D[i, j] = kl_divergence(reduction_distributions[i], reduction_distributions[j], 0)

        # 3) Convert hospitalization_risks to a probability distribution
        risk_prob = hospitalization_risks / (hospitalization_risks.sum() + 1e-12)

        # 4) Construct the pair‑wise risk‑difference matrix
        H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))  # element (i,j) = risk_prob[i] - risk_prob[j]

        H_d_MAX = H_d.max()
        H_d_MIN = H_d.min()

        KL_D_scaled = ((KL_D - KL_D_MIN) / (KL_D_MAX - KL_D_MIN)) * (H_d_MAX - H_d_MIN) + H_d_MIN

        # 6) Aggregate difference between scaled KL and risk differences (off‑diagonal only)
        mask_off_diag = ~np.eye(K, dtype=bool)
        fairness = np.sum(np.abs(KL_D_scaled[mask_off_diag] - H_d[mask_off_diag])) / ((K * K) - K)

    return fairness * (-1)


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
    #plt.ylim(-1.16, -1.14)
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
            y_values = [cumulative_reward[6]]
        elif fairness == "abfta":
            y_values = [cumulative_reward[7]]
        elif fairness == "sb_sbs":
            y_values = [cumulative_reward[5], cumulative_reward[6]]
        elif fairness == "sbs_abfta":
            y_values = [cumulative_reward[6], cumulative_reward[7]]
        else:
            y_values = [cumulative_reward[5]]

        policy_results.append([hospitalizations] + y_values)

    return np.array(policy_results)



def make_test_reduction_matrices():
    mats = {"fair, same risk same distribution": np.array([
        [0.2, 0.0, 0.3, 0.2, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.3, 0.2, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.3, 0.0, 0.5, 0.1, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.4, 0.4],
        [0.1, 0.3, 0.0, 0.5, 0.1, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.4, 0.4]
    ]), "unfair, same risks different distribution": np.array([
        [0.2, 0.0, 0.3, 0.2, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.7, 0.1, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.1, 0.3, 0.0, 0.5, 0.1, 0.0],
        [0.4, 0.1, 0.3, 0.1, 0.1, 0.0],
        [0.1, 0.3, 0.0, 0.5, 0.1, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.4, 0.4]
    ]),
        "unfair, one reduction": np.array([
            [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1, 0.0, 0.0, 0.0, 0.0],
            [1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ])
    }

    return mats


def run_fairness_on_test_matrices():
    """Compute the fairness metric for each crafted matrix."""
    mats = make_test_reduction_matrices()
    risk = np.array([0.07977, 0.02279, 0.04558, 0.03704, 0.06553, 0.08262, 0.11966, 0.21083, 0.12536, 0.21083])
    risk_prob = risk / risk.sum()

    print("\n=== Manual scenario fairness checks ===")
    for name, red in mats.items():
        reduction_per_age_group = red

        # 1) Normalize reduction_per_age_group so each of the 10 rows sums to 1
        reduction_sums = reduction_per_age_group.sum(axis=1, keepdims=True)
        reduction_sums[reduction_sums == 0] = 1e-12
        reduction_distributions = reduction_per_age_group / reduction_sums

        # 2) Construct pair‑wise KL‑divergence matrix
        K = reduction_distributions.shape[0]
        KL_D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                KL_D[i, j] = hellinger(reduction_distributions[i], reduction_distributions[j], 0)

        # 3) Convert hospitalization_risks to a probability distribution

        KL_D_MAX = 28
        KL_D_MIN = 0

        # 4) Construct the pair‑wise risk‑difference matrix
        H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))  # element (i,j) = risk_prob[i] - risk_prob[j]

        H_d_MAX = H_d.max()
        H_d_MIN = H_d.min()

        KL_D_scaled = ((KL_D - KL_D_MIN) / (KL_D_MAX - KL_D_MIN)) * (H_d_MAX - H_d_MIN) + H_d_MIN

        # 6) Aggregate difference between scaled KL and risk differences (off‑diagonal only)
        mask_off_diag = ~np.eye(K, dtype=bool)
        fairness = np.sum(np.abs(KL_D_scaled[mask_off_diag] - H_d[mask_off_diag]))
        print(f"Scenario '{name}': fairness = {fairness:.4f}")
    print("=== End checks ===\n")


def env_random_demo():
    import gym
    from itertools import product

    from fairness.individual.individual_fairness import get_reduction_impact

    env = gym.make("BECovidWithLockdownODEBudget2Continuous-v0")

    np.set_printoptions(precision=5, suppress=True, linewidth=400)
    n_samples = 1000000

    # run our fixed scenario tests once

    run_fairness_on_test_matrices()
    max_KL = 0
    max_f = 0
    min_f = 100000
    step = 0
    values = np.round(np.linspace(0.0, 1.0, 80), decimals=2)
    # actions = np.array(list(product(values, repeat=3)))
    i = 0
    while True:
        env.reset()
        done = False
        while not done:
            step += 1
            a = env.action_space.sample()
            obs, reward, done, info = env.step(a)

            hospitalization_risks = env.model.get_hospitalization_risk()
            reduction_per_age_group = np.abs(get_reduction_impact(env.C_diff_fairness))

            # 1) Normalize reduction_per_age_group so each of the 10 rows sums to 1
            reduction_sums = reduction_per_age_group.sum(axis=1, keepdims=True)
            reduction_sums[reduction_sums == 0] = 1e-12
            reduction_distributions = reduction_per_age_group / reduction_sums

            # 2) Construct pair‑wise KL‑divergence matrix
            K = reduction_distributions.shape[0]
            KL_D = np.zeros((K, K))
            for i in range(K):
                for j in range(K):
                    KL_D[i, j] = kl_divergence(reduction_distributions[i], reduction_distributions[j], 0)

            # 3) Convert hospitalization_risks to a probability distribution
            risk_prob = hospitalization_risks / (hospitalization_risks.sum() + 1e-12)

            # 4) Construct the pair‑wise risk‑difference matrix
            H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))  # element (i,j) = risk_prob[i] - risk_prob[j]

            H_d_MAX = H_d.max()
            H_d_MIN = H_d.min()

            KL_D_scaled = ((KL_D - KL_D_MIN) / (KL_D_MAX - KL_D_MIN)) * (H_d_MAX - H_d_MIN) + H_d_MIN

            # 6) Aggregate difference between scaled KL and risk differences (off‑diagonal only)
            mask_off_diag = ~np.eye(K, dtype=bool)
            fairness = np.sum(np.abs(KL_D_scaled[mask_off_diag] - H_d[mask_off_diag])) / ((K * K) - K)

            print(f"Action: {a} --- Step: {step} --- Fairness: {fairness}")
            if fairness > max_f:
                max_f = fairness
            if fairness < min_f:
                min_f = fairness

            step += 1


def plot_bars(actions_ex_extreme, abfta_values_extreme, label):
    plt.figure(figsize=(15, 10))
    plt.bar(actions_ex_extreme, abfta_values_extreme, label='ABFTA for different actions', color=label_colors[label.lower()])
    plt.xticks(rotation=45, ha='right')

    plt.xlabel('Action')
    plt.ylabel('Age-based Fairness Through Unawareness')
    plt.title(f'Age-based Fairness Through Unawareness for {label} actions')
    plt.legend()
    plt.ylim(0, 0.09)
    plt.show()


#
# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute KL‑divergence PDF without plotting
def compute_kl_pdf(states, bins=50):
    """Return (bin_centers, pdf_vals) for aggregated scaled KL across all states."""
    KL_D_MAX = 28
    KL_D_MIN = 0

    all_kl_vals = []
    for state_df, C_diff in states:
        hosp_risk = state_df["h_risk"].values
        red = np.abs(get_reduction_impact(C_diff))

        # Normalise each row
        row_sums = red.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-12
        red_dist = red / row_sums

        K = red_dist.shape[0]
        KL_D = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                KL_D[i, j] = kl_divergence(red_dist[i], red_dist[j], 0)

        risk_prob = hosp_risk / (hosp_risk.sum() + 1e-12)
        H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))

        H_d_MAX, H_d_MIN = H_d.max(), H_d.min()
        KL_scaled = ((KL_D - KL_D_MIN) / (KL_D_MAX - KL_D_MIN)) * (H_d_MAX - H_d_MIN) + H_d_MIN
        all_kl_vals.extend(KL_scaled[~np.eye(K, dtype=bool)])

    pdf_vals, bin_edges = np.histogram(all_kl_vals, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, pdf_vals


# ──────────────────────────────────────────────────────────────────────────────
def inspect_abfta_kl(env, save_dir=None):
    """
    For 'Extreme', 'Fixed' and 'Random' action sets:
      • Collect ABFTA for each action.
      • Collect KL‑divergence values for the same states.
      • Show a single figure with two sub‑plots: bar chart (ABFTA) and PDF (KL).
    """
    print("Inspecting ABFTA and KL divergence")
    set_params_policies()
    # --- Action lists ---------------------------------------------------------
    extreme_actions = [np.array(a, dtype=np.float32)
                       for a in itertools.product([0., 0.5, 1.], repeat=3)]

    n_fixed = len(extreme_actions)                       # keep counts identical
    fixed_actions = [np.array([u, u, u], dtype=np.float32)
                     for u in np.linspace(0., 1., n_fixed)]

    random_actions = [env.action_space.sample() for _ in range(n_fixed)]

    action_sets = {"Constrasting": extreme_actions,
                   "Fixed":   fixed_actions,
                   "Random":  random_actions}
    set_params_policies()
    # --- Evaluate each set ----------------------------------------------------
    for label, actions in action_sets.items():
        abfta_vals, action_labels, kl_states = [], [], []

        env.reset()                                      # fresh run for this set
        for act in actions:
            env.reset()
            _, _, _, _ = env.step(act)
            _, _, _, _ = env.step(act)
            abfta_vals.append(abs(compute_abfta([env.state_df()])))
            action_labels.append(str(np.round(act, 2)))
            kl_states.append(env.state_df())

        # Compute KL PDF once per set
        kl_x, kl_pdf = compute_kl_pdf(kl_states)

        # ------------------------ plotting ------------------------------------
        fig, (bar_ax, kl_ax) = plt.subplots(1, 2, figsize=(16, 7))
        # Same styling as hospitalization bar plot
        bar_ax.set_axisbelow(True)
        bar_ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        bar_ax.spines['top'].set_visible(False)
        bar_ax.spines['right'].set_visible(False)
        kl_ax.set_axisbelow(True)
        kl_ax.grid(True, linestyle='--', alpha=0.7)
        kl_ax.spines['top'].set_visible(False)
        kl_ax.spines['right'].set_visible(False)

        # Bar chart (ABFTA)
        bars = bar_ax.bar(action_labels, abfta_vals, edgecolor='gray',
                   color=label_colors[label.lower()], label="ABFTA", zorder=2)
        # Value annotations
        for bar in bar_ax.patches:
            h = bar.get_height()
            bar_ax.annotate(f'{h:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7)
        bar_ax.set_xlabel("Action", fontsize=12)
        bar_ax.set_ylabel("Age-based Fairness Through Unawareness", fontsize=12)
        bar_ax.set_title(f"Age-based Fairness Through Unawareness – {label} actions", fontsize=14, fontweight='normal', pad=10)

        # Force the ticks to match your actions, rotate & align them
        bar_ax.set_xticks(range(len(action_labels)))
        bar_ax.set_xticklabels(
            action_labels,
            rotation=45,
            ha='right',
            fontsize=10
        )
        bar_ax.tick_params(axis='x', pad=3)

        plt.subplots_adjust(bottom=0.25)
        # Move legend above the bars
        # bar_ax.legend(
        #     loc='upper center',
        #     bbox_to_anchor=(0.85, 1),
        #     ncol=1,
        #     frameon=False,
        #     fontsize=10
        # )

        # KL PDF
        kl_ax.plot(kl_x, kl_pdf,
                   color=label_colors[label.lower()], zorder=2, linewidth=3)
        kl_ax.set_xlabel("Scaled pairwise KL‑divergence", fontsize=12)
        kl_ax.set_ylabel("Probability density", fontsize=12)
        kl_ax.set_title(f"KL divergence PDF – {label} actions", fontsize=14, fontweight='normal', pad=10)
        kl_ax.set_ylim(0, 80)

        fig.suptitle(f"{label} actions: Fairness & Social Restriction divergence", fontsize=14)
        # leave extra room at top for legend
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # Save or show
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"abfta_kl_{label}.jpg"), dpi=dpi)
        else:
            plt.show()


def inspect_sbs(env, save_dir=None):
    """
    For 'Extreme', 'Fixed' and 'Random' action sets:
      • Collect Social Burden Score (SBS) for each action.
      • Show a bar chart of SBS values.
    """
    print("Inspecting Social Burden Fairness")
    set_params_policies()
    # --- Action lists -----------------------------------------------------
    extreme_actions = [np.array(a, dtype=np.float32)
                       for a in itertools.product([0., 0.5, 1.], repeat=3)]
    n_fixed = len(extreme_actions)
    fixed_actions = [np.array([u, u, u], dtype=np.float32)
                     for u in np.linspace(0., 1., n_fixed)]
    random_actions = [env.action_space.sample() for _ in range(n_fixed)]
    test_actions = [np.array([1, 1, 1])]

    action_sets = {
        #"test": test_actions,
        "Constrasting": extreme_actions,
        "Fixed":   fixed_actions,
        "Random":  random_actions
    }

    # --- Evaluate each set ------------------------------------------------
    for label, actions in action_sets.items():
        sbs_vals, action_labels = [], []
        for act in actions:
            env.reset()
            _, _, _, _ = env.step(act)
            _, _, _, _ = env.step(act)
            # compute Social Burden Fairness for the current state
            val = compute_sbs([env.state_df()])
            sbs_vals.append(abs(val))
            action_labels.append(str(np.round(act, 2)))

        # --- plotting SBS bar chart ----------------------------------------
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_axisbelow(True)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        bars = ax.bar(action_labels,
               sbs_vals, edgecolor='gray',
               color=label_colors[label.lower()],
               label="SBF", zorder=2)
        # Value annotations
        for bar in ax.patches:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=6)
        ax.set_xlabel("Action", fontsize=12)
        ax.set_ylabel("Social Burden Fairness", fontsize=12)
        ax.set_title(f"Social Burden Fairness – {label} actions", fontsize=14, fontweight='normal', pad=10)
        ax.set_xticklabels(
            action_labels,
            rotation=45,
            ha='right',
            fontsize=12
        )
        ax.tick_params(axis='x', pad=3)
        ax.set_ylim(0, 25000)
        #ax.legend()
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"sbs_{label}.jpg"), dpi=dpi)
        else:
            plt.show()


def inspect_prop_lost(env, fairness, save_dir=None):
    """
    Evaluate proportional lost contacts for four specific actions:
        [1,0,1], [1,1,0], [0,1,1], [1,1,1]

    Produces a 2×2 grid of bar charts (one per action), with one bar
    per age-group showing the proportion of contacts lost.
    """
    print(f"Inspecting lost contacts for {fairness}")
    set_params_policies()
    import matplotlib.pyplot as plt
    import numpy as np

    bar_color = "mediumpurple"
    # hospitalization risks scaled by 10
    hosp_vals = hospitalization_risks * 10

    # --- action list -------------------------------------------------------
    if fairness == "abfta":
        actions = [
            np.array([1., 0., 1.], dtype=np.float32),
            np.array([1., 1., 0.], dtype=np.float32),
            np.array([0., 1., 0.], dtype=np.float32),
            np.array([0., 1., 1.], dtype=np.float32),
        ]
        action_labels = ["[w: 1, s: 0, l: 1]", "[w: 1, s: 1, l: 0]", "[w: 0, s: 1, l: 0]", "[w: 0, s: 1, l: 1]"]
    else:
        actions = [
            np.array([0., 0., 0.], dtype=np.float32),
            np.array([0., 0., 1.], dtype=np.float32),
            np.array([0., 1., 0.], dtype=np.float32),
            np.array([0., 1., 1.], dtype=np.float32),
        ]
        action_labels = ["[w: 0, s: 0, l: 0]", "[w: 0, s: 0, l: 1]", "[w: 0, s: 1, l: 0]", "[w: 0, s: 1, l: 1]"]
    # --- collect data ------------------------------------------------------
    age_labels = None
    prop_lost_list = []   # one entry per action
    fairness_list = []
    h_list = []
    kl_list = []

    for act in actions:
        env.reset()
        if fairness == "abfta":
            _, _, _, info = env.step(act)
            _, _, _, info = env.step(act)
            prop_lost = info["prop_lost_contacts_per_age"]
            prop_lost_list.append(prop_lost)
            fval = compute_abfta([env.state_df()])
            fairness_list.append(abs(fval))
            # compute average scaled KL divergence for this state
            state_df, C_diff = env.state_df()
            hosp = state_df["h_risk"].values
            red = np.abs(get_reduction_impact(C_diff))
            # normalize rows
            row_sums = red.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-12
            red_dist = red / row_sums
            K = red_dist.shape[0]
            KL_D = np.zeros((K, K))
            for i_kl in range(K):
                for j_kl in range(K):
                    KL_D[i_kl, j_kl] = kl_divergence(red_dist[i_kl], red_dist[j_kl], 0)
            risk_prob = hosp / (hosp.sum() + 1e-12)
            H_d = np.abs(np.subtract.outer(risk_prob, risk_prob))
            KL_scaled = ((KL_D - 0) / (28 - 0)) * (H_d.max() - H_d.min()) + H_d.min()
            avg_kl = np.mean(KL_scaled[~np.eye(K, dtype=bool)])
            kl_list.append(avg_kl)
            h_list.append(np.mean(H_d[~np.eye(K, dtype=bool)]))
        else:
            _, _, _, info = env.step(act)
            _, _, _, info = env.step(act)
            prop_lost = info["prop_lost_contacts_per_age"]
            prop_lost_list.append(prop_lost)
            fval = compute_sbs([env.state_df()])
            fairness_list.append(abs(fval))

        # grab the age-group labels once
        if age_labels is None:
            age_labels = env.state_df()[0].index.tolist()

    # --- plot 2×2 grid -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    # draw grid behind bars and style axes
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if fairness == "abfta":
        for idx, (ax, props, lbl, fval, hval, kval) in enumerate(zip(
                axes, prop_lost_list, action_labels, fairness_list, h_list, kl_list)):
            # plot lost contacts and hospitalization risks side by side
            x_age = np.arange(len(age_labels))
            width = 0.4

            # lost contacts bars
            lost_bars = ax.bar(x_age - width/2, props, width=width,
                               color=bar_color, edgecolor='gray', zorder=2)
            # hospitalization risk bars
            hosp_bars = ax.bar(x_age + width/2, hosp_vals, width=width,
                               color='skyblue', edgecolor='gray', zorder=1)

            # annotate lost-contact bars
            for bar in lost_bars:
                h = bar.get_height()
                label_text = '0' if h == 0 else f'{h:.2f}'
                ax.annotate(label_text,
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(-2, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6)
            # annotate hospitalization risk bars
            for bar in hosp_bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(1, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6)

            ax.set_ylim(0, 1)
            # secondary axis for KL and fairness
            ax2 = ax.twinx()
            # plot average KL (to the left)
            x_h = len(age_labels)
            ax2.bar(x_h, hval, color='lightpink', width=0.5, zorder=2, edgecolor='grey')
            ax2.annotate(f'{hval:.2f}',
                         xy=(x_h, hval),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', color='black', fontsize=7)
            # plot average KL (to the left)
            x_kl = len(age_labels) + 1
            ax2.bar(x_kl, kval, color='mediumvioletred', width=0.5, zorder=2, edgecolor='grey')
            ax2.annotate(f'{kval:.2f}',
                         xy=(x_kl, kval),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', color='black', fontsize=7)
            # plot fairness (to the right)
            x_f = len(age_labels) + 2
            ax2.bar(x_f, fval, color='plum', edgecolor='gray', width=0.5, zorder=2)
            ax2.annotate(f'{fval:.2f}',
                         xy=(x_f, fval),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', color='black', fontsize=7)
            ax2.set_ylim(0, 0.11)
            # labels and ticks
            labels_ext = age_labels + ["d̅", "D̅", "Unfairness"]
            ax.set_xticks(list(x_age) + [x_h, x_kl, x_f])
            ax.set_xticklabels(labels_ext, rotation=45, ha='right', fontsize=9)
            # reduce gap between tick labels and axis
            ax.tick_params(axis='x', pad=3)

            ax.set_xlabel("Age Group", fontsize=12)
            ax.set_ylabel("Proportionally Lost Contacts", fontsize=12)
            ax2.set_ylabel("Value", color='black', fontsize=12)
            ax.set_title(f"Proportional lost contacts – action {lbl}", fontsize=14, fontweight='normal', pad=10)

            # add a shared legend beneath the title
            handles = [
                mpatches.Patch(color='mediumpurple', label='Proportionally Lost Contacts'),
                mpatches.Patch(color='skyblue', label='Hospitalization Risk (×10)'),
                mpatches.Patch(color='lightpink', label='Average Difference in Hospitalization Risk'),
                mpatches.Patch(color='mediumvioletred', label='Average Scaled KL divergence'),
                mpatches.Patch(color='plum', label='Unfairness'),
            ]
            fig.legend(handles=handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.95),
                       ncol=5,
                       frameon=False,
                       fontsize=10)

    else:
        for idx, (ax, props, lbl, fval) in enumerate(zip(
                axes, prop_lost_list, action_labels, fairness_list)):
            # plot lost contacts and hospitalization risks side by side
            x_age = np.arange(len(age_labels))
            width = 0.4

            # lost contacts bars
            lost_bars = ax.bar(x_age - width/2, props, width=width,
                               color=bar_color, edgecolor='gray', zorder=2)
            # hospitalization risk bars
            hosp_bars = ax.bar(x_age + width/2, hosp_vals, width=width,
                               color='skyblue', edgecolor='gray', zorder=1)

            # annotate lost-contact bars
            for bar in lost_bars:
                h = bar.get_height()
                label_text = '0' if h == 0 else f'{h:.2f}'
                ax.annotate(label_text,
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(-1, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6)
            # annotate hospitalization risk bars
            for bar in hosp_bars:
                h = bar.get_height()
                ax.annotate(f'{h:.2f}',
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(1, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=6)

            ax.set_ylim(0, 1)
            # secondary axis for fairness
            ax2 = ax.twinx()
            # plot fairness (to the right)
            x_f = len(age_labels)
            ax2.bar(x_f, fval, color='plum', edgecolor='gray', width=0.5, zorder=2)
            ax2.annotate(f'{fval:.2f}',
                         xy=(x_f, fval),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', color='black', fontsize=7)

            # labels and ticks
            labels_ext = age_labels + ["Unfairness"]
            ax.set_xticks(list(x_age) + [x_f])
            ax.set_xticklabels(labels_ext, rotation=45, ha='right', fontsize=9)
            # reduce gap between tick labels and axis
            ax.tick_params(axis='x', pad=3)
            ax2.set_ylim(0, 25000)

            ax.set_xlabel("Age Group", fontsize=12)
            ax.set_ylabel("Proportionally Lost Contacts", fontsize=12)
            ax2.set_ylabel("Value", color='black', fontsize=12)
            ax.set_title(f"Proportional lost contacts – action {lbl}", fontsize=14, fontweight='normal', pad=10)

            # add a shared legend beneath the title
            handles = [
                mpatches.Patch(color='mediumpurple', label='Proportionally Lost Contacts'),
                mpatches.Patch(color='skyblue', label='Hospitalization Risk (×10)'),
                mpatches.Patch(color='plum', label='Unfairness'),
            ]
            fig.legend(handles=handles,
                       loc='upper center',
                       bbox_to_anchor=(0.5, 0.95),
                       ncol=3,
                       frameon=False,
                       fontsize=10)

    if fairness == "abfta":
        label = "Age-based Fairness Through Unawareness"
    else:
        label = "Social Burden Fairness"
    plt.suptitle(f"Proportionally lost contacts per age group for actions considering {label}",
                 fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"prop_lost_contacts_{fairness}.jpg"), dpi=dpi)
    else:
        plt.show()


def inspect_total_contacts(env, hosp=False, save_dir=None):
    """
    Simulate one episode with action [1, 1, 1] and visualise the
    *aggregated* total‑contact matrices for Work, School and Leisure
    as three heat‑maps in a single figure.

    Parameters
    ----------
    env : gym.Env
        The epidemic environment.
    hosp : bool, default False
        If *True* the matrices are divided element‑wise by the
        hospitalization risks to give “contacts per unit risk”.
    save_dir : str or Path or None
        Directory in which to save the figure.  If *None* the file is
        created in the current working directory.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    action = np.array([1., 1., 1.], dtype=np.float32)

    env.reset()
    done = False
    K = env.K

    # accumulators for the episode
    work_mat    = np.zeros((K, K))
    school_mat  = np.zeros((K, K))
    leisure_mat = np.zeros((K, K))

    while not done:
        _, _, done, info = env.step(action)
        work_mat    += info["total_contacts_matrix_work"]
        school_mat  += info["total_contacts_matrix_school"]
        leisure_mat += info["total_contacts_matrix_leisure"]

        if hosp:
            risk_mat = hospitalization_risks[:, None]  # shape (K,1) broadcast
            work_mat    = work_mat / risk_mat
            school_mat  = school_mat / risk_mat
            leisure_mat = leisure_mat / risk_mat

    # ---- plotting ----------------------------------------------------
    mats  = [work_mat, school_mat, leisure_mat]
    titles = ["Work", "School", "Leisure"]

    vmin = min(mat.min() for mat in mats)
    vmax = max(mat.max() for mat in mats)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, mat, title in zip(axes, mats, titles):
        im = ax.imshow(mat, cmap="magma", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Age group j")
        ax.set_ylabel("Age group i")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))

    # shared colour bar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Total contacts", rotation=-90, va="bottom")

    sup_title = ("Total contacts matrices – action [w:1, s:1, l:1]"
                 if not hosp else
                 "Contacts per hospitalization‑risk – action [w:1, s:1, l:1]")
    fig.suptitle(sup_title, fontsize=16)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # ---- save --------------------------------------------------------
    fname = "total_contacts_matrices.jpg" if not hosp else "total_contacts_matrices_hosp.jpg"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)


def inspect_hospitalization_differences(save_dir=None):
    """
    Compute the pair‑wise absolute difference matrix H_d of
    age‑specific hospitalization risks and save it as a heat‑map
    (JPEG, dpi = 300).

    Parameters
    ----------
    save_dir : str or Path or None
        Directory in which to write the figure.  If *None* the file
        is created in the current working directory.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) pair‑wise absolute differences -------------------------------
    H_d = np.abs(np.subtract.outer(hospitalization_risks,
                                   hospitalization_risks))

    # 2) plotting ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H_d, cmap=heatmap_color, origin="upper")

    # axis styling
    ax.set_xlabel("Age-group")
    ax.set_ylabel("Age-group")
    ax.set_title("Pair‑wise absolute differences in Hospitalization Risks", pad=12)
    age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90+']
    ax.set_xticks(range(len(age_labels)))
    ax.set_yticks(range(len(age_labels)))
    ax.set_xticklabels(age_labels, rotation=45, ha='right')
    ax.set_yticklabels(age_labels)

    # colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Absolute difference", rotation=-90, va="bottom")

    plt.tight_layout()

    for i in range(len(H_d)):
        rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                 linewidth=0,
                                 edgecolor=None,
                                 facecolor=diagonal_color,
                                 zorder=3)
        ax.add_patch(rect)

    # 3) save ----------------------------------------------------------
    fname = "hospitalization_risk_diff_heatmap.jpg"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, fname)
    fig.savefig(fname, dpi=dpi)
    plt.close(fig)

def total_contacts_per_area(env, hosp=False, save_dir=None):
    """
    Simulate a single episode with the maximal action [1, 1, 1].
    Aggregate the total contacts for each area (school, work, leisure)
    across *all* timesteps and plot them per age‑group.

    The bar‑chart layout mimics `inspect_sbs`: three side‑by‑side bars
    for every age‑group with distinct colours.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    print("Inspecting total contacts (school / work / leisure)")
    set_params_policies()

    # -------------- run one episode ------------------------------------
    action = np.array([1., 1., 1.], dtype=np.float32)
    env.reset()
    done = False

    K = env.K
    total_school   = np.zeros(K)
    total_work     = np.zeros(K)
    total_leisure  = np.zeros(K)


    _, _, done, info = env.step(action)
    total_school  += info["total_contacts_school"]
    total_work    += info["total_contacts_work"]
    total_leisure += info["total_contacts_leisure"]

    if hosp:
        total_school /= hospitalization_risks
        total_work /= hospitalization_risks
        total_leisure /= hospitalization_risks
    # -------------- plotting ------------------------------------------
    age_labels = env.state_df()[0].index.tolist()
    x = np.arange(K)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    # Styling (same as inspect_sbs)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    bars_school  = ax.bar(x - width, total_school,
                          width=width, color=action_colors["s"],
                          edgecolor='gray', label='School')
    bars_work    = ax.bar(x         , total_work,
                          width=width, color=action_colors["w"],
                          edgecolor='gray', label='Work')
    bars_leisure = ax.bar(x + width, total_leisure,
                          width=width, color=action_colors["l"],
                          edgecolor='gray', label='Leisure')

    # Annotate bars with value
    for bars in (bars_school, bars_work, bars_leisure):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Total contacts", fontsize=12)
    if hosp:
        ax.set_title("Average total contacts per Age‑group per Area divided by Hospitalization risk",
                     fontsize=14, fontweight='normal', pad=10)
    else:
        ax.set_title("Average total contacts per Age‑group per Area",
                     fontsize=14, fontweight='normal', pad=10)
    ax.set_xticks(x)
    ax.tick_params(axis='x', pad=3)
    ax.set_xticklabels(age_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(0.90, 0.90))

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if hosp:
            fig.savefig(os.path.join(save_dir, "total_contacts_area_hosp.jpg"), dpi=dpi)
        else:
            fig.savefig(os.path.join(save_dir, "total_contacts_area.jpg"), dpi=dpi)
    else:
        plt.show()


if __name__ == '__main__':

    # env_random_demo()
    y_measure = "abfta"
    env_type = "ODE"
    budget = 5
    if budget == None:
        env = gym.make(f'BECovidWithLockdown{env_type}Continuous-v0')
    else:
        env = gym.make(f'BECovidWithLockdown{env_type}Budget{budget}Continuous-v0')

    save_dir = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/"

    inspect_hospitalization_differences(save_dir=save_dir)
    
    #inspect_total_contacts(env, hosp=True, save_dir=save_dir)
    # total_contacts_per_area(env, hosp=True, save_dir=save_dir)

    inspect_abfta_kl(env, save_dir=save_dir)
    # #
    # inspect_prop_lost(env, "abfta", save_dir=save_dir)
    # #
    inspect_sbs(env, save_dir=save_dir)
    # #
    # inspect_prop_lost(env, "sbs", save_dir=save_dir)

    # coverage_set = generate_fixed_coverage_set(env, y_measure, amount_of_policies=100)
    #
    # # Create a DataFrame and save as a CSV file:
    # if coverage_set.shape[1] == 2:
    #     df = pd.DataFrame(coverage_set, columns=["hospitalizations", "measure"])
    # else:
    #     df = pd.DataFrame(coverage_set, columns=["hospitalizations", "measure1", "measure2"])
    # df.to_csv(f"fixed_policy_{y_measure}_new.csv", index=False)
    # print(f"Saved fixed policies in fixed_policy_{y_measure}.csv")
    # plot_coverage_set([f"fixed_policy_{y_measure}.csv"], y_measure)
    # plot_coverage_set(["testcov.csv", "fixed_sb.csv"], y_measure)
    # plot_coverage_set(["window17.csv", f"fixed_{y_measure}.csv"], "SBS")
