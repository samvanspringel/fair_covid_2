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
import h5py
import pathlib

# Same imports as eval_pcn, except we replace the environment creation
# with your create_fair_env (create_fair_covid_env).
# Adjust paths as needed:
from scenario.create_fair_env import create_fair_covid_env
from agent.pcn.pcn import non_dominated, Transition, choose_action, epsilon_metric

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

MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/experiments/results/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/ref_results/seed_0/6obj_3days_crashed/model_9.pt"  # your single best model


def interpolate_runs(runs, w=100):
    # Create a common x_grid from all runs’ first column values
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:, 0] for r in runs]))))
    all_values = np.stack([np.interp(all_steps, r[:, 0], r[:, 1]) for r in runs], axis=0)
    return all_steps, all_values

def load_runs_from_logdir(logdir):
    """
    Recursively finds all 'log.h5' files in logdir, loads the final Pareto front
    from each, and returns a list of dicts (each has 'pareto_front').
    """
    logdir_path = Path(logdir)
    runs = []
    for path in logdir_path.rglob('log.h5'):
        with h5py.File(path, 'r') as logfile:
            pf = logfile['train/leaves/ndarray']
            pareto_front = pf[-1]  # final front
            runs.append({'pareto_front': pareto_front})
    return runs

def plot_fixed_data():
    """
    Reads in fixed.csv (assumes two columns: x and y) and plots its data using plt.scatter.
    """
    df_fixed = pd.read_csv("fixed.csv")
    plt.scatter(df_fixed["o_0"], df_fixed["o_1"], s=5, alpha=0.7, label="fixed", marker='o')

def plot_pareto_front_from_dir(logdir, budget_label, scale_x=10000, scale_y=100,
                               save=True, use_interpolation=True, extreme_y_threshold=-20):
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
    # 1) Load all runs
    runs = load_runs_from_logdir(logdir)

    # 2) Process each run’s Pareto front:
    sorted_runs = []
    for run in runs:
        pf = run['pareto_front']
        pf = np.unique(pf, axis=0)
        pf = non_dominated(pf)
        # Filter out points with y (second column) below the threshold.
        pf = pf[pf[:, 1] >= extreme_y_threshold]
        if pf.size == 0:
            continue  # Skip this run if all points are filtered out
        pf = pf[np.argsort(pf[:, 0])]  # sort by first dimension (objective 0)
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
        plt.plot(x_vals_scaled, mean_curve, label=f'Mean coverage set (b={budget_label})')
        plt.fill_between(x_vals_scaled, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
        # Overlay fixed.csv data.
        plot_fixed_data()

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
            plt.plot(x, y, alpha=0.7, label='Seed curve')
        # Overlay fixed.csv data.
        plot_fixed_data()

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

def make_budget_plots():
    # The top-level folder containing budget_{i} subdirs
    BASELINE_DIR = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/agent/pcn/baseline_results"
    plt.rcParams["figure.figsize"] = (15, 15)
    # Suppose you have budgets 0..5:
    all_budgets = [0, 2, 3, 4, 5]
    #all_budgets = [2, 3, 4, 5]
    # We'll store the (x_vals, mean_curve, std_curve) in a dict
    results_dict = {}

    # Step A: Generate one figure per budget
    for b in all_budgets:
        subdir = os.path.join(BASELINE_DIR, f"budget_{b}")
        x_vals_scaled, mean_curve, std_curve = plot_pareto_front_from_dir(
            logdir=subdir,
            budget_label=str(b),
            scale_x=10000,
            scale_y=100,
            save=True
        )
        if x_vals_scaled is not None:
            # Store for final combined plot
            results_dict[b] = (x_vals_scaled, mean_curve, std_curve)

    # Step B: Combine all budgets into one single figure
    plt.figure()
    for b, (x_vals_scaled, mean_curve, std_curve) in results_dict.items():
        plt.plot(x_vals_scaled, mean_curve, label=f'Budget={b}')
        plt.fill_between(
            x_vals_scaled,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.1
        )
    plot_fixed_data()
    plt.xlabel('Hospitalizations (scaled)')
    plt.ylabel('Social Burden (scaled)')
    plt.title('Coverage set variation for multiple budgets')
    plt.legend()
    plt.tight_layout()
    plt.savefig("NCS_ref_interp_all_budgets.png")
    print("Saved NCS_ref_interp_all_budgets.png")
    plt.show()

def plot_pareto_fronts_sbs():
    csv_files = ["sbs.csv", "sbs1.csv", "sbs2.csv", "sbs3.csv", "sbs4.csv"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))

    df_fixed = pd.read_csv("fixed.csv")
    axs[0].scatter(df_fixed["o_0"], df_fixed["o_1"], s=5, alpha=0.7, label="fixed", marker='o')

    arh_l, arh_max, arh_min = [], [], []
    sb_l, sb_max, sb_min = [], [], []
    sbs_l, sbs_max, sbs_min = [], [], []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        arh = df["o_0"]
        arh_l.append(arh.mean())
        arh_max.append(arh.max())
        arh_min.append(arh.min())

        sb = df["o_1"]
        sb_l.append(sb.mean())
        sb_max.append(sb.max())
        sb_min.append(sb.min())

        sbs = df["o_2"]
        sbs_l.append(sbs.mean())
        sbs_max.append(sbs.max())
        sbs_min.append(sbs.min())

        axs[0].scatter(df["o_0"], df["o_1"], s=5, alpha=0.7, label=csv_file, marker='o')
        axs[1].scatter(df["o_0"], df["o_2"], s=5, alpha=0.7, label=csv_file, marker='o')

    print(f"arh mean of (Min, Mean, Max) :", (np.array(arh_min).mean(), np.array(arh_l).mean(), np.array(arh_max).mean()))
    print(f"sb mean of (Min, Mean, Max) :", (np.array(sb_min).mean(), np.array(sb_l).mean(), np.array(sb_max).mean()))
    print("sbs mean of (Min, Mean, Max) :", (np.array(sbs_min).mean(), np.array(sbs_l).mean(), np.array(sbs_max).mean()))

    axs[0].set_ylim(-2000, 0)
    axs[0].set_xlabel("Hospitalizations")
    axs[0].set_ylabel("Social burden")
    axs[0].set_title("NCS: Hospitalizations vs Social Burden")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_xlabel("Hospitalizations")
    axs[1].set_ylabel("Social burden score")
    axs[1].set_title("NCS: Hospitalizations vs Social Burden Score")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("NCS_sbs.png")

def plot_pareto_fronts_abfta():
    csv_files = ["abfta.csv", "abfta1.csv", "abfta2.csv", "abfta3.csv", "abfta4.csv"]
    fig, axs = plt.subplots(2, 1, figsize=(10, 14))

    df_fixed = pd.read_csv("fixed.csv")
    axs[0].scatter(df_fixed["o_0"], df_fixed["o_1"], s=5, alpha=0.7, label="fixed", marker='o')

    arh_l, arh_max, arh_min = [], [], []
    sb_l, sb_max, sb_min = [], [], []
    abfta_l, abfta_max, abfta_min = [], [], []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        arh = df["o_0"]
        arh_l.append(arh.mean())
        arh_max.append(arh.max())
        arh_min.append(arh.min())

        sb = df["o_1"]
        sb_l.append(sb.mean())
        sb_max.append(sb.max())
        sb_min.append(sb.min())

        abfta = df["o_2"]
        abfta_l.append(abfta.mean())
        abfta_max.append(abfta.max())
        abfta_min.append(abfta.min())

        axs[0].scatter(df["o_0"], df["o_1"], s=5, alpha=0.7, label=csv_file, marker='o')
        axs[1].scatter(df["o_0"], df["o_2"], s=5, alpha=0.7, label=csv_file, marker='o')

    print(f"arh mean of (Min, Mean, Max) :", (np.array(arh_min).mean(), np.array(arh_l).mean(), np.array(arh_max).mean()))
    print(f"sb mean of (Min, Mean, Max) :", (np.array(sb_min).mean(), np.array(sb_l).mean(), np.array(sb_max).mean()))
    print("abfta mean of (Min, Mean, Max) :", (np.array(abfta_min).mean(), np.array(abfta_l).mean(), np.array(abfta_max).mean()))

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
    plt.savefig("NCS_abfta.png")


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

        # Multiply each slice of C_diff by term_matrix and sum over i and j
        fairness = np.sum(C_diff * term_matrix, axis=(1, 2))
        fairness_window += fairness.sum()
    return fairness_window

def compute_abfta(states):
    fairness_window = 0

    for state_df, C_diff in states:
        reduction_impact = get_reduction_impact(C_diff)
        hospitalization_risks = state_df["h_risk"].values
        K = len(hospitalization_risks)

        fairness = 0
        n = 0
        for i in range(K):
            for j in range(i + 1, K):
                distance_reductions = get_distance_reduction(reduction_impact, i, j)
                diff = np.abs(hospitalization_risks[i] - hospitalization_risks[j]) - distance_reductions
                fairness += diff
                n += 1

        if n > 0:
            fairness_window += -1 + (fairness / n)

    return fairness_window

def run_episode(env, model, desired_return, desired_horizon, max_return, objectives, fn):
    """
    Identical to eval_pcn.py's run_episode.
    """
    transitions = []
    obs = env.reset()
    done = False
    lost_contacts_per_agegroup = []
    fairness_states = []

    abfta_states = []

    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, eval=True)
        n_obs, reward, done, info = env.step(action)
        fairness_states.append(env.state_df())
        if len(objectives) > 2:
            extra = len(objectives) - len(reward)
            filler = np.empty(extra)
            reward = np.append(reward, filler)

        transitions.append(
            Transition(
                observation=obs[1],
                action=action,
                reward=np.float32(reward).copy(),
                next_observation=n_obs[1],
                terminal=done
            )
        )
        lost_contacts_per_agegroup.append(info["lost_contacts_per_age"])

        obs = n_obs
        # clip desired return to avoid negative or beyond max
        desired_return = np.clip(desired_return - reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative
        desired_horizon = np.float32(max(desired_horizon - 1, 1.0))

    if fn == "sbs":
        sbs_states = compute_sbs(fairness_states)
    elif fn == "abfta":
        abfta_states = compute_abfta(fairness_states)
    elif fn == "both":
        sbs_states = compute_sbs(fairness_states)
        abfta_states = compute_abfta(fairness_states)

    return transitions, lost_contacts_per_agegroup, sbs_states, abfta_states


def plot_episode(transitions, lost_contacts, alpha=1.0):
    fig = plt.figure(figsize=(14, 12), constrained_layout=True)
    gs = fig.add_gridspec(6, 1)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[3:, 0])
    ]

    states = np.array([t.observation for t in transitions])
    # add final state
    states = np.concatenate((states, transitions[-1].next_observation[None]), axis=0)
    ari = (states[:-1, :, 0] - states[1:, :, 0]).sum(axis=-1)
    i_hosp_new = states[..., -3].sum(axis=-1)
    i_icu_new = states[..., -2].sum(axis=-1)
    d_new = states[..., -1].sum(axis=-1)
    actions = np.array([t.action for t in transitions])
    # append action of None
    actions = np.concatenate((actions, [[None] * 3]))

    # steps in dates
    start = datetime.date(2020, 5, 3)
    week = datetime.timedelta(days=7)
    dates = [start + week * i for i in range(0, 18, 2)]

    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp_new, alpha=alpha, label='hosp', color='blue')
    ax.plot(i_icu_new, alpha=alpha, label='icu', color='green')
    ax.plot(i_hosp_new + i_icu_new, label='hosp+icu', alpha=alpha, color='orange')
    ax.set_xticks(ticks=np.arange(0, 18, 2), labels=[str(d.day) + '/' + str(d.month) for d in dates])

    # deaths
    ax = axs[1]
    ax.plot(d_new, alpha=alpha, label='deaths', color='red')
    # ax.plot(ari, alpha=alpha, label='ari', color='black')

    # actions
    ax = axs[2]
    ax.set_ylim([0, 1])
    ax.plot(actions[:, 0], alpha=alpha, label='p_w', color='blue')
    ax.plot(actions[:, 1], alpha=alpha, label='p_s', color='orange')
    ax.plot(actions[:, 2], alpha=alpha, label='p_l', color='green')

    axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')
    axs[1].set_ylabel('deaths')
    axs[2].set_ylabel('actions')
    # for ax in axs:
    #     ax.legend()

    # Use the 4th axis (axs[3]) to plot the lost contacts arrays
    ax = axs[3]
    lost_contacts_array = np.stack(lost_contacts, axis=0)  # shape (timesteps, 10)

    for i in range(lost_contacts_array.shape[1]):
        ax.plot(lost_contacts_array[:, i], label=f'Age {age_groups[i]}')

    ax.set_ylabel('Lost Contacts')
    ax.set_title('Lost Contacts per Age Group')
    ax.legend()

    return [start + week * i for i in range(0, 18, 1)], ari, i_hosp_new, i_icu_new, d_new, actions[:, 0], actions[:,
                                                                                                          1], actions[:,
                                                                                                              2]


def evaluate_pcn(env, model, desired_return, desired_horizon, max_return, objectives, results_title, fn, gamma=1.0, n=1):
    alpha = 1 if n == 1 else 0.2
    returns = np.empty((n, desired_return.shape[-1]))
    all_transitions = []
    for n_i in range(n):
        transitions, lost_contacts = run_episode(env, model, desired_return, desired_horizon, max_return, objectives, fn)
        # compute return
        for i in reversed(range(len(transitions) - 1)):
            transitions[i].reward += gamma * transitions[i + 1].reward

        returns[n_i] = transitions[i].reward.flatten()
        print(f'ran model with desired-return: {desired_return.flatten()}, got {transitions[i].reward.flatten()}')
        print('action sequence: ')
        for t in transitions:
            print(f'- {t.action}')
        t = plot_episode(transitions, lost_contacts, alpha)
        t = t + tuple(zip(*[ti.reward * env.scale for ti in transitions]))

        df = pd.DataFrame([x for x in zip(*t)],
                          columns=['dates', 'ari', 'i_hosp_new', 'i_icu_new', 'd_new', 'p_w', 'p_s', 'p_l'] + [f'o_{oi}'
                                                                                                               for oi in
                                                                                                               range(
                                                                                                                   returns.shape[
                                                                                                                       1])])
        # manually set p_s to 0 during school holidays
        holidays = df['dates'] >= datetime.date(2020, 7, 1)
        df.loc[holidays, 'p_s'] = 0
        all_transitions.append(df)
    title = results_title + ': Re: ' + ' '.join([f'{o:.3f}' for o in (returns.mean(0) * env.scale)[objectives]])
    title += '\n'
    title += results_title + ': Rt: ' + ' '.join([f'{o:.3f}' for o in (desired_return * env.scale)[objectives]])
    plt.suptitle(title)
    print(
        f'ran model with desired-return: {desired_return[objectives].flatten()}, got average return {returns[:, objectives].mean(0).flatten()}')
    return returns, all_transitions


def main():
    """
    Reproduces eval_pcn.py logic but uses create_fair_covid_env from create_fair_env.py
    to build the environment, and then loads & evaluates a model from the specified dir.
    """
    # ------------------------------------------------------------
    # 1) Define arguments that we would normally parse
    # ------------------------------------------------------------
    class Args:
        seed = 0
        env = 'covid'
        episode_length = 100
        bias = 0
        ignore_sensitive = False
        budget = 5
        # Add any others you need from create_fair_covid_env
        # ...
    args = Args()



    # ------------------------------------------------------------
    # 3) Hardcode the path to your logs/model
    # ------------------------------------------------------------
    model_paths_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/agent/pcn/"
    model_paths = {"ref_results" : [0, 1],
                   "sbs_results" : [0, 1, 2],
                   "abfta_results" : [0, 1, 2],
                   "sbs_abfta_results" : [0, 1, 2, 3]}

    fn = ["none", "sbs", "abfta", "both"]

    index = 0
    for path, obj in model_paths.items():
        fn = fn[index]
        index += 1

        # ------------------------------------------------------------
        # 2) Create the environment from create_fair_env
        # ------------------------------------------------------------
        env, scale, ref_point, scaling_factor, max_return, *_ = create_fair_covid_env(
            args, rewards_to_keep=obj
        )
        print(scale)
        print(scaling_factor)
        print(max_return)
        print("Environment created:", env)

        results_path = model_paths_dir + path

        model_dir = pathlib.Path(results_path)
        print("Using model directory:", model_dir)

        # Load the HDF5 log if needed
        log_h5 = model_dir / "log.h5"
        # This is how eval_pcn does it:
        with h5py.File(log_h5, 'r') as log:
            # read the final pareto front
            pf_array = log['train/leaves/ndarray'][-1]
            # get nondominated front, just like eval_pcn
            pf, pf_i = non_dominated(pf_array[:, obj], return_indexes=True)  # example if you had objectives 1,5
            # sort them
            pf_sorted = pf[ np.argsort(pf, axis=0)[:, 0] ]
            print("Pareto front shape:", pf_sorted.shape)

        # ------------------------------------------------------------
        # 4) Load the most recent checkpoint model_10.pt (or whichever)
        # ------------------------------------------------------------
        # Typically, eval_pcn uses `model_10.pt`; you can adapt if needed:
        checkpoints = sorted([str(p) for p in model_dir.glob('model_10.pt')])
        if len(checkpoints) == 0:
            raise FileNotFoundError("No model_*.pt found in directory.")
        print("Found checkpoint(s):", checkpoints)
        model = torch.load(checkpoints[-1], map_location=device, weights_only=False)
        model.eval()

        # ------------------------------------------------------------
        # 5) Evaluate
        #    We'll replicate the typical usage: pick a 'desired_return'
        #    from the pareto front, or define your own. We'll choose
        #    the 0th item from pf_sorted as an example, or a direct
        #    desired_return array. Also define `objectives`.
        # ------------------------------------------------------------
        objectives = obj      # match your environment's indexing
        desired_return = pf_sorted[0]  # e.g. first solution from the front
        desired_return = desired_return.astype(np.float32)
        desired_horizon = 17.0

        # Evaluate the model on 1 or more episodes
        evaluate_pcn(
            env,
            model,
            desired_return,
            desired_horizon,
            max_return,
            objectives,
            path,
            fn,
            gamma=1.0,
            n=1,
        )
        plt.savefig(path)



if __name__ == "__main__":
    make_budget_plots()
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # plot_pareto_fronts_sbs()
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # plot_pareto_fronts_abfta()
    #main()
