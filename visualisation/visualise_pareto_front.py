from pathlib import Path
from agent.pcn.pcn import choose_action
from scenario.create_fair_env import *
from scenario.pcn_model import *
from fairness.individual.individual_fairness import *

import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import ast
import h5py
import pathlib
from agent.pcn.eval_pcn import eval_pcn

# Same imports as eval_pcn, except we replace the environment creation
# with your create_fair_env (create_fair_covid_env).
# Adjust paths as needed:
from scenario.create_fair_env import create_fair_covid_env
from agent.pcn.pcn import non_dominated, Transition, choose_action, epsilon_metric
plt.rcParams['legend.fontsize']    = 24 #14   # legend text Trajectories
plt.rcParams['axes.titlesize']     = 28 #16   # axes & figure titles Trajectories
plt.rcParams['axes.labelsize']     = 24 #14   # x/y axis labels Trajectories
plt.rcParams['xtick.labelsize']    = 20 #12   # x-tick numbers Trajectories
plt.rcParams['ytick.labelsize']    = 20 #12   # y-tick numbers Trajectories

age_groups = {
    0: "0-10",
    1: "10-20",
    2: "20-30",
    3: "30-40",
    4: "40-50",
    5: "50-60",
    6: "60-70",
    7: "70-80",
    8: "80-90",
    9: "90-..."
}
phi0 = np.array([0.972, 0.992, 0.984, 0.987, 0.977, 0.971, 0.958, 0.926, 0.956, 0.926])
delta2_star = 0.756
hospitalization_risks = (1 - phi0) * delta2_star

MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/experiments/results/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/ref_results/seed_0/6obj_3days_crashed/model_9.pt"  # your single best model


def interpolate_runs(runs, w=100):
    # Create a common x_grid from all runs’ first column values
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:, 0] for r in runs]))))
    all_values = np.stack([np.interp(all_steps, r[:, 0], r[:, 1]) for r in runs], axis=0)
    return all_steps, all_values


def load_runs_from_logdir(logdir, objectives, objectives_plot):
    """
    Recursively finds all 'log.h5' files in logdir, loads the final Pareto front
    from each, and returns a list of dicts (each has 'pareto_front').
    """
    logdir_path = Path(logdir)
    runs = []
    for path in logdir_path.rglob('log.h5'):
        print(path)
        with h5py.File(path, 'r') as logfile:
            pareto_front = logfile['train/leaves/ndarray'][-1]
            _, pareto_front_i = non_dominated(pareto_front[:, objectives], return_indexes=True)
            pareto_front = pareto_front[pareto_front_i]

            pf = np.argsort(pareto_front, axis=0)
            pareto_front = pareto_front[pf[:, 0]]
            pareto_front = pareto_front[:, objectives_plot]
            runs.append({'pareto_front': pareto_front})
    return runs


def plot_fixed_data(measure):
    """
    Reads in fixed.csv (assumes two columns: x and y) and plots its data using plt.scatter.
    """
    df_fixed = pd.read_csv(f"fixed_policy_{measure}.csv")
    # Check if mean of first column > 3000, and scale if needed
    if 'o_0' in df_fixed.columns and 'o_1' in df_fixed.columns:
        x = df_fixed['o_0'].values
        y = df_fixed['o_1'].values
    else:
        x = df_fixed.iloc[:, 0].values
        y = df_fixed.iloc[:, 1].values
    plt.scatter(x, y, s=50, alpha=0.7, edgecolors='k', label="fixed", marker='o')


def plot_pareto_front_from_dir(measure, measure_to_plot, logdir, budget_label, scale_x=10000, scale_y=90,
                               save=False, use_interpolation=True, extreme_y_threshold=-20):
    """
    1) Loads Pareto‐front data from `logdir`.
    2) Removes duplicate points, computes the nondominated front, and filters out extreme y values (< extreme_y_threshold).
    3) If use_interpolation is True:
          - Interpolates across seeds and computes the mean ± std.
       Otherwise:
          - Plots each seed’s curve directly (raw curves without interpolation).
    4) In both cases, overlays the data from fixed.csv.
    5) Finally, it plots and (optionally) saves the resulting figure.
    Returns (x_vals_scaled, mean_curve, std_curve) when using interpolation; otherwise (None, None, None).
    """
    if measure == "sbs":
        objectives = [1, 6]
    elif measure == "sb_sbs":
        objectives = [1, 5, 6]
    elif measure == "abfta":
        objectives = [1, 7]
    elif measure == "sbs_abfta":
        objectives = [1, 6, 7]
    else:
        objectives = [1, 5]

    if measure_to_plot == "sbs":
        objectives_plot = [1, 6]
    elif measure_to_plot == "abfta":
        objectives_plot = [1, 7]
    else:
        objectives_plot = [1, 5]
    runs = load_runs_from_logdir(logdir, objectives, objectives_plot)

    # 2) Process each run’s Pareto front:
    sorted_runs = []
    for run in runs:
        pf = run['pareto_front']
        pf = np.unique(pf, axis=0)
        # pf = non_dominated(pf)
        # Filter out points with y (second column) below the threshold.
        # pf = pf[pf[:, 1] >= extreme_y_threshold]
        if pf.size == 0:
            continue  # Skip this run if all points are filtered out
        sorted_runs.append(pf)

    if not sorted_runs:
        print(f"No valid runs found for {logdir} after filtering extreme y values.")
        return None, None, None

    if use_interpolation:
        # Interpolate across seeds on a common grid.
        x_vals, y_vals = interpolate_runs(sorted_runs)
        # Scale the x and y values.
        x_vals_scaled = x_vals * scale_x
        y_vals_scaled = y_vals * scale_y
        # Compute mean and standard deviation across runs.
        mean_curve = np.mean(y_vals_scaled, axis=0)
        std_curve = np.std(y_vals_scaled, axis=0)

        plt.figure()
        plt.plot(x_vals_scaled, mean_curve, linewidth=2, label=f'Mean coverage set (b={budget_label})')
        plt.fill_between(x_vals_scaled, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        # Overlay fixed.csv data.
        plot_fixed_data(measure_to_plot)

        plt.xlabel('Hospitalizations (scaled)')
        plt.ylabel('Social Burden (scaled)')
        plt.title(f'Coverage set variation (budget={budget_label})')
        plt.legend()
        plt.tight_layout()
        if save:
            outname = f"NCS_ref_interp_budget_{budget_label}.png"
            plt.savefig(outname)
            print(f"Saved {outname}")
        else:
            plt.show()
        return x_vals_scaled, mean_curve, std_curve
    else:
        # Plot raw curves for each seed.
        plt.figure()
        for pf in sorted_runs:
            x = pf[:, 0] * scale_x
            y = pf[:, 1] * scale_y
            plt.plot(x, y, linewidth=2, alpha=0.7, label='Seed curve')
        # Overlay fixed.csv data.
        plot_fixed_data(measure_to_plot)

        plt.xlabel('Hospitalizations (scaled)')
        plt.ylabel('Social Burden (scaled)')
        plt.title(f'Coverage set variation (raw data) (budget={budget_label})')
        plt.legend()
        plt.tight_layout()
        if save:
            outname = f"NCS_nointerp_budget_{budget_label}.png"
            plt.savefig(outname)
            print(f"Saved {outname}")
        else:
            plt.show()
        return None, None, None


def make_budget_plots(measure, measure_to_plot, scale_x, scale_y):
    # The top-level folder containing budget_{i} subdirs
    BASELINE_DIR = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/{measure}_results"
    plt.rcParams["figure.figsize"] = (17, 15)
    # Suppose you have budgets 0..5:
    all_budgets = [0, 2, 3, 4, 5]
    # We'll store the (x_vals, mean_curve, std_curve) in a dict
    results_dict = {}

    # Step A: Generate one figure per budget
    for b in all_budgets:
        subdir = os.path.join(BASELINE_DIR, f"budget_{b}")
        x_vals_scaled, mean_curve, std_curve = plot_pareto_front_from_dir(
            measure,
            measure_to_plot,
            logdir=subdir,
            budget_label=str(b),
            scale_x=scale_x,
            scale_y=scale_y,
            save=False
        )
        if x_vals_scaled is not None:
            # Store for final combined plot
            results_dict[b] = (x_vals_scaled, mean_curve, std_curve)

    # Step B: Combine all budgets into one single figure
    plt.figure()
    for b, (x_vals_scaled, mean_curve, std_curve) in results_dict.items():
        if b == 0:
            b_label = "∞"
        else:
            b_label = b
        plt.plot(x_vals_scaled, mean_curve, linewidth=3, label=f'Budget={b_label}')
        plt.fill_between(
            x_vals_scaled,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.1
        )
    plot_fixed_data(measure_to_plot)
    plt.xlabel('Hospitalizations ')
    plt.ylabel(f'{measure_to_plot}')
    plt.title('Pareto front')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("NCS_ref_interp_all_budgets.png")
    print("Saved NCS_ref_interp_all_budgets.png")
    plt.show()


device = 'cpu'


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


def compute_abfta(states):
    fairness_window = 0
    for state_df, C_diff in states:
        reduction_impact = get_reduction_impact(C_diff)  # shape (K, 6)
        risks = state_df["h_risk"].values  # shape (K,)
        # compute pairwise risk-normalized distances
        D = risk_normalized_distance_matrix(reduction_impact, risks, metric="euclidean")
        # accumulate fairness as sum of distances (zero means perfect proportionality)
        fairness_window += np.sum(D)
    return fairness_window * (-1)


def evaluate_pcn(measure, model_dir, objectives, save_dir):
    n_budget = 5
    budget = f'Budget{n_budget}' if n_budget is not None else ''
    model_dir = pathlib.Path(model_dir)
    save_dir = pathlib.Path(save_dir)
    # Recursively find the log.h5 file under model_dir
    log_files = list(model_dir.rglob('log.h5'))
    if not log_files:
        raise FileNotFoundError(f"No log.h5 found under {model_dir}")
    log_path = log_files[0]
    log = h5py.File(log_path, 'r')

    # Recursively find all model checkpoint files under model_dir
    checkpoint_paths = list(model_dir.rglob('model_*.pt'))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No model_*.pt files found under {model_dir}")
    checkpoint_paths = sorted(checkpoint_paths)
    model = torch.load(str(checkpoint_paths[-1]), weights_only=False)

    with log:
        pareto_front = log['train/leaves/ndarray'][-1]
        _, pareto_front_i = non_dominated(pareto_front[:, objectives], return_indexes=True)
        pareto_front = pareto_front[pareto_front_i]

        pf = np.argsort(pareto_front, axis=0)
        pareto_front = pareto_front[pf[:, 0]]

    env_type = 'ODE'

    scale, ref_point, scaling_factor, max_return = get_scaling()

    core_objectives_max = 6

    scale = scale[0:core_objectives_max]
    scaling_factor = scaling_factor[0:core_objectives_max]
    ref_point = ref_point[0:core_objectives_max]
    # max_return = max_return[0:core_objectives_max]

    envs = {}
    env = gym.make(f'BECovidWithLockdown{env_type}{budget}Continuous-v0')
    # env = gym.make(f'BECovidWithLockdownUntil2021{env_type}Continuous-v0')
    nA = np.prod(env.action_space.shape)
    env.action = lambda x: x
    env = TodayWrapper(env)
    env = ScaleRewardEnv(env, scale=scale)
    print(env)

    interactive = False

    inp = -1
    if not interactive:
        (model_dir / 'policies-executions').mkdir(exist_ok=True)
        print('=' * 38)
        print('not interactive, this may take a while')
        print('=' * 38)
        all_returns = []
    while True:
        inp = inp + 1
        if inp >= len(pareto_front):
            break
        desired_return = pareto_front[inp]
        desired_return = desired_return
        print(f"Lengths: {len(scale)}, {len(max_return)}, {len(desired_return)}")

        desired_horizon = 17  # 35

        r, t = eval_pcn(env, model, desired_return, desired_horizon, max_return, objectives, measure)
        # plt.savefig(model_dir / 'policies-executions' / f'policy_{inp}.png')
        for i, t_i in enumerate(t):
            (save_dir / 'policies-transitions' / f'{inp}').mkdir(exist_ok=True, parents=True)
            t_i.to_csv(save_dir / 'policies-transitions' / f'{inp}' / f'run_{i}.csv', index_label='index')
        all_returns.append(r)


import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def plot_policies(policy_ids, data_dir):
    figsize_x = 13
    figsize_y = 3 * len(policy_ids)
    figsize = (figsize_x, figsize_y)

    fig, axes = plt.subplots(nrows=len(policy_ids), ncols=1, sharex=True, figsize=figsize)
    date_fmt = DateFormatter("%d/%m")

    for ax, pid in zip(axes, policy_ids):
        # find the run CSV inside the numbered policy folder
        if pid == -1:
            subdirs = [
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
            ]
            if not subdirs:
                raise FileNotFoundError(f"No numeric policy subdirectories in {data_dir}")
            last = str(max(int(d) for d in subdirs))
            print(f"Using last policy folder: {last}")
            policy_folder = os.path.join(data_dir, last)
        else:
            print(pid)
            policy_folder = os.path.join(data_dir, str(pid))
        run_files = glob.glob(os.path.join(policy_folder, 'run*.csv'))
        if not run_files:
            raise FileNotFoundError(f"No run CSV file found in {policy_folder}")
        csv_path = run_files[0]
        df = pd.read_csv(csv_path, parse_dates=['dates'])

        # left axis
        ax.plot(df['dates'], df['i_hosp_new'], color='deepskyblue', label='daily new hosp')
        ax.plot(df['dates'], df['i_icu_new'], color='orange', label='daily new ICU')
        ax.plot(df['dates'], df['d_new'], color='red', label='daily deaths')
        ax.set_ylabel('individuals')
        ax.set_ylim(0, 4500)
        ax.tick_params(axis='y', labelcolor='black')

        # right axis
        ax2 = ax.twinx()
        ax2.plot(df['dates'], df['p_w'], '--', color='blue', label='$p_w$')
        ax2.plot(df['dates'], df['p_s'], '--', color='magenta', label='$p_s$')
        ax2.plot(df['dates'], df['p_l'], '--', color='black', label='$p_l$')
        ax2.set_ylabel('proportion')
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis='y', labelcolor='black')

        # formatting
        ax.xaxis.set_visible(False)  # hide all x‐ticks until the bottom plot
        ax2.xaxis.set_visible(False)

    # bottom plot gets the time axis
    axes[-1].xaxis.set_visible(True)
    axes[-1].set_xlabel('time')
    axes[-1].xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()

    # single legend at top
    # collect all handles & labels from the top subplot
    handles = []
    labels = []
    for axis in (axes[0], axes[0].twinx()):
        h, l = axis.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc='upper center', ncol=6, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # make room for the legend
    plt.show()

    # Also plot lost contacts per age group for each policy
    plot_lost_contacts_per_age(policy_ids, data_dir)


# New function: plot_lost_contacts_per_age
def plot_lost_contacts_per_age(policy_ids, data_dir):
    """
    Plot the evolution of lost contacts per age group over time for each policy.
    """
    age_groups = {
        0: "0-10",
        1: "10-20",
        2: "20-30",
        3: "30-40",
        4: "40-50",
        5: "50-60",
        6: "60-70",
        7: "70-80",
        8: "80-90",
        9: "90-..."
    }
    import json
    for pid in policy_ids:
        # handle -1 as "last" numeric subfolder
        if pid == -1:
            subdirs = [
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
            ]
            if not subdirs:
                raise FileNotFoundError(f"No numeric policy subdirectories in {data_dir}")
            last = str(max(int(d) for d in subdirs))
            print(f"Using last policy folder for lost contacts: {last}")
            policy_folder = os.path.join(data_dir, last)
        else:
            policy_folder = os.path.join(data_dir, str(pid))
        run_files = glob.glob(os.path.join(policy_folder, 'run*.csv'))
        if not run_files:
            print(f"No run CSV file found in folder {policy_folder}")
            continue
        df = pd.read_csv(
            run_files[0],
            parse_dates=['dates'],
            converters={'lost_contacts': json.loads}
        )
        if 'lost_contacts' in df.columns:
            lost_list = df['lost_contacts'].tolist()
        else:
            print(f"No lost contacts column found in CSV for policy {pid}")
            continue
        lost_arr = np.vstack(lost_list)  # shape (T, age_groups)
        dates = df['dates']
        plt.figure(figsize=(13, 4))
        for i, label in age_groups.items():
            plt.plot(dates, lost_arr[:, i], label=label)
        plt.xlabel('Date')
        plt.ylabel('Lost contacts per age group')
        plt.title(f'Policy {pid} - Lost Contacts per Age Group Over Time')
        plt.legend(loc='upper right', ncol=5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------- #
# New helper: combined trajectory + lost‑contacts plot for a single policy id #
# --------------------------------------------------------------------------- #
def plot_policy_with_contacts(pid, data_dir, measure, show_lost_contacts=True):
    """
    Create a single figure with two rows:
        1) epidemic trajectory (daily hosp/ICU/deaths + p_w, p_s, p_l)
        2) lost contacts per age group
    """
    import json
    from matplotlib.dates import DateFormatter

    # Resolve policy folder (handle -1 as "last"):
    if pid == -1:
        subdirs = [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
        ]
        if not subdirs:
            raise FileNotFoundError(f"No numeric policy subdirectories in {data_dir}")
        last = str(max(int(d) for d in subdirs))
        print(f"Using last policy folder: {last}")
        policy_folder = os.path.join(data_dir, last)
    else:
        last = pid
        policy_folder = os.path.join(data_dir, str(pid))

    run_files = glob.glob(os.path.join(policy_folder, 'run*.csv'))
    if not run_files:
        raise FileNotFoundError(f"No run CSV file found in {policy_folder}")
    csv_path = run_files[0]

    df = pd.read_csv(
        csv_path,
        parse_dates=['dates'],
        converters={'lost_contacts': json.loads}
    )

    # choose subplot layout
    if show_lost_contacts:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]}
        )
    else:
        fig, ax_top = plt.subplots(1, 1, figsize=(13, 5), sharex=True)
    date_fmt = DateFormatter("%d/%m")

    # --- Top axis: trajectory ------------------------------------------------
    ax_top.plot(df['dates'], df['i_hosp_new'], color='deepskyblue', label='daily new hosp')
    ax_top.plot(df['dates'], df['i_icu_new'], color='orange', label='daily new ICU')
    ax_top.plot(df['dates'], df['d_new'], color='red', label='daily deaths')
    ax_top.set_ylabel('individuals')
    ax_top.set_ylim(0, 4500)

    ax2 = ax_top.twinx()
    ax2.plot(df['dates'], df['p_w'], '--', color='blue', label='$p_w$')
    ax2.plot(df['dates'], df['p_s'], '--', color='magenta', label='$p_s$')
    ax2.plot(df['dates'], df['p_l'], '--', color='black', label='$p_l$')
    ax2.set_ylabel('proportion')
    ax2.set_ylim(-0.05, 1.05)

    # Combine legends from both y‑axes
    handles, labels = [], []
    for axis in (ax_top, ax2):
        h, l = axis.get_legend_handles_labels()
        handles += h
        labels += l
    ax_top.legend(handles, labels, loc='upper left', ncol=4, frameon=False)

    # --- Bottom axis: lost contacts -----------------------------------------
    if show_lost_contacts:
        if 'lost_contacts' in df.columns:
            lost_arr = np.vstack(df['lost_contacts'].tolist())  # shape (T, 10)
            age_groups = {
                0: "0-10", 1: "10-20", 2: "20-30", 3: "30-40", 4: "40-50",
                5: "50-60", 6: "60-70", 7: "70-80", 8: "80-90", 9: "90‑..."
            }
            for i, label in age_groups.items():
                ax_bot.plot(df['dates'], lost_arr[:, i], label=label)
            ax_bot.set_ylabel('lost contacts')
            ax_bot.set_ylim(-20, 0)
            ax_bot.legend(loc='upper right', ncol=5, frameon=False)
        else:
            ax_bot.text(0.5, 0.5, "No lost_contacts column in CSV", transform=ax_bot.transAxes,
                        ha='center', va='center')

        ax_bot.set_xlabel('date')
        ax_bot.xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate()

    plt.suptitle(f'Policy {last} for {measure}')
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# New convenience wrapper to call the above for multiple policies            #
# --------------------------------------------------------------------------- #
def plot_multiple_policies_with_contacts(policy_ids, data_dir, measure):
    """
    Iterate over the given list of policy_ids and call plot_policy_with_contacts
    for each, producing separate figures.
    """
    for pid in policy_ids:
        plot_policy_with_contacts(pid, data_dir, measure, show_lost_contacts=False)


def select_policies_by_total_hospitalizations(transitions_dir, target, tol=300, obj_col="o_1"):
    """
    Return subfolder IDs whose final 'o_0' value is within ±tol of target.
    """
    selected = []
    for d in os.listdir(transitions_dir):
        if not d.isdigit():
            continue
        csv_path = os.path.join(transitions_dir, d, "run_0.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, parse_dates=["dates"])
        if obj_col not in df.columns:
            raise KeyError(f"Expected column '{obj_col}' in {csv_path}")
        total_hosp = df["i_hosp_new"].sum()
        if (np.abs(total_hosp) <= target + tol and np.abs(total_hosp) >= target - tol):
            print(total_hosp)
            selected.append(int(d))
    return min(selected, key=lambda x: abs(x - target))


def main(measure, goal_hospitalizations, run_episodes=False):
    if measure == "sbs":
        objectives = [1, 6]
    elif measure == "sb_sbs":
        objectives = [1, 5, 6]
    elif measure == "abfta":
        objectives = [1, 7]
    elif measure == "sbs_abfta":
        objectives = [1, 6, 7]
    else:
        objectives = [1, 5]

    budget = 5
    seed = 2

    model_paths_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results"

    model_subdir = f"{measure_pcn}_results"
    inspect_policies = [-1]
    budget_str = f"budget_{budget}"

    seed_str = f"seed_{seed}"

    model_dir = os.path.join(model_paths_dir, model_subdir, budget_str, seed_str)

    transitions_dir = os.path.join(model_paths_dir, model_subdir, budget_str, "policies-transitions")

    if run_episodes:
        evaluate_pcn(measure, model_dir, objectives, os.path.join(model_paths_dir, model_subdir, budget_str))
    else:
        # plot_policies(inspect_policies, transitions_dir)
        inspect_policies = select_policies_by_total_hospitalizations(transitions_dir,
                                                                     target=goal_hospitalizations,
                                                                     tol=300)
        plot_multiple_policies_with_contacts([inspect_policies], transitions_dir, measure)
        return inspect_policies


def plot_age_group_comparison(pids, age_group_idx, measures, model_paths_dir, budget):
    """
    Compare lost contacts over time for a specific age group across multiple measures.
    """
    import json
    from matplotlib.dates import DateFormatter

    plt.figure(figsize=(12, 6))
    date_fmt = DateFormatter("%d/%m")

    for i, measure in enumerate(measures):
        # build path to policies-transitions for this measure
        transitions_base = os.path.join(model_paths_dir,
                                        f"{measure}_results",
                                        f"budget_{budget}",
                                        "policies-transitions")
        # find last numeric policy folder
        subdirs = [
            d for d in os.listdir(transitions_base)
            if os.path.isdir(os.path.join(transitions_base, d)) and d.isdigit()
        ]
        if not subdirs:
            print(f"No policies found for measure {measure} in {transitions_base}")
            continue
        if pids[i] == -1:
            id = str(max(int(d) for d in subdirs))
        else:
            id = str(pids[i])
        policy_folder = os.path.join(transitions_base, id)

        run_files = glob.glob(os.path.join(policy_folder, 'run_*.csv'))
        if not run_files:
            print(f"No run CSV found for measure {measure} in {policy_folder}")
            continue

        # load the first run's data
        df = pd.read_csv(run_files[0],
                         parse_dates=['dates'],
                         converters={'lost_contacts': json.loads})
        lost_arr = np.abs(np.vstack(df['lost_contacts'].tolist()))
        dates = df['dates']

        plt.plot(dates,
                 lost_arr[:, age_group_idx],
                 label=measure)

    plt.xlabel('Date')
    plt.ylabel(f"Lost contacts (age {age_groups.get(age_group_idx)})")
    plt.title(f"Lost contacts over time for age group {age_groups.get(age_group_idx)}")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_age_group_grid_comparison(pids, measures, model_paths_dir, budget):
    """
    Plot lost contacts over time for each age group in a 5×2 grid:
    Left column: age groups [0,6,7,8,9]; Right column: [1,2,3,4,5].
    """
    import json
    from matplotlib.dates import DateFormatter

    right_ids = [0, 6, 7, 8, 9]
    left_ids = [1, 2, 3, 4, 5]
    age_cols = [left_ids, right_ids]

    fig, axes = plt.subplots(5, 2, figsize=(40, 40), sharex=True)
    date_fmt = DateFormatter("%d/%m")

    for col, age_list in enumerate(age_cols):
        for row, age_idx in enumerate(age_list):
            ax = axes[row, col]
            for pid, measure in zip(pids, measures):
                base = os.path.join(
                    model_paths_dir,
                    f"{measure}_results",
                    f"budget_{budget}",
                    "policies-transitions"
                )
                subdirs = [
                    d for d in os.listdir(base)
                    if os.path.isdir(os.path.join(base, d)) and d.isdigit()
                ]
                if not subdirs:
                    raise FileNotFoundError(f"No numeric subdirs in {base}")
                folder = (
                    os.path.join(base, str(max(int(d) for d in subdirs)))
                    if pid == -1 else os.path.join(base, str(pid))
                )
                run_files = glob.glob(os.path.join(folder, "run_*.csv"))
                if not run_files:
                    print(f"No run CSV for {measure} in {folder}")
                    continue
                df = pd.read_csv(
                    run_files[0], parse_dates=["dates"], converters={"lost_contacts": json.loads}
                )
                lost_arr = np.abs(np.vstack(df["lost_contacts"].tolist()))
                dates = df["dates"]
                ax.plot(dates, lost_arr[:, age_idx], label=measure)

            ax.set_ylabel(age_groups[age_idx])
            if row == 0 and col == 0:
                ax.legend(loc="upper right", ncol=len(measures))
            ax.xaxis.set_major_formatter(date_fmt)

    # label shared x-axis on bottom row
    for col in (0, 1):
        axes[-1, col].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# Example:
# plot_age_group_grid_comparison(pids, all_measures, model_paths_dir, budget)


def plot_hospitalization_risk_bar(age_groups_dict, risk_array):
    """
    Plot a bar chart of hospitalization risk per age group.

    :param age_groups_dict: dict mapping age_index to label (e.g. {0: '0-10', ...})
    :param risk_array: 1D array‐like of risk values (must align with the keys/order of age_groups_dict)
    """
    # Sort by age index to get consistent ordering
    indices = sorted(age_groups_dict.keys())
    labels = [age_groups_dict[i] for i in indices]
    values = [risk_array[i] for i in indices]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xlabel('Age Group')
    plt.ylabel('Hospitalization Risk')
    plt.title('Hospitalization Risk by Age Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def get_scaling_plot(measure):
    if measure == "sb":
        return 10000, 90
    elif measure == "sbs":
        return 10000, 24e4
    elif measure == "abfta":
        return 10000, 5


if __name__ == "__main__":
    all_measures = ["sb"]#, "sbs", "abfta", "sb_sbs", "sbs_abfta"]
    model_paths_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results"
    budget = 5
    #plot_hospitalization_risk_bar(age_groups, hospitalization_risks)
    measure_pcn = "sbs"

    measure_to_plot = "sb"
    scale_x, scale_y = get_scaling_plot(measure_to_plot)
    make_budget_plots(measure_pcn, measure_to_plot, scale_x, scale_y)

    # run_episodes = False
    # pids = []
    # goal_hospitalizations = 8000
    # for measure_pcn in all_measures:
    #     print(f"Finding policy for {measure_pcn}")
    #     policy_id = main(measure_pcn, goal_hospitalizations, run_episodes)
    #     pids.append(policy_id)
    #
    # # for i in range(len(age_groups)):
    # #     plot_age_group_comparison(pids, i, all_measures, model_paths_dir, budget)
    #
    # plot_age_group_grid_comparison(pids, all_measures, model_paths_dir, budget)
