from pathlib import Path

from matplotlib import patches

from agent.pcn.pcn import choose_action
from scenario.create_fair_env import *
from scenario.pcn_model import *
from fairness.individual.individual_fairness import *
import pylab
from visualisation.vis.visualise import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D surface plots
import datetime
import pandas as pd
import ast
import h5py
import pathlib
from agent.pcn.eval_pcn import eval_pcn

# Optional: Plotly for interactive 3D plots
try:
    import plotly.graph_objects as go
except ImportError:
    go = None  # Plotly optional

# Same imports as eval_pcn, except we replace the environment creation
# with your create_fair_env (create_fair_covid_env).
# Adjust paths as needed:
from scenario.create_fair_env import create_fair_covid_env
from agent.pcn.pcn import non_dominated, Transition, choose_action, epsilon_metric

budget_colors = {2: "dodgerblue", 3: "darkorange", 4: "tomato", 5: "mediumpurple", 0: "forestgreen"}
objectives_labels = {1: "Hospitalizations", 5: "Social Burden", 6: "Social Burden Fairness", 7: "Age-Based FTU"}
measure_mapping = {
    "sb": "Social Burden",
    "sb_sbs": "Social Burden and Social Burden Fairness",
    "sbs": "Social Burden Fairness",
    "abfta": "Age-based FTU",
    "sbs_abfta": "Social Burden Fairness and Age-based FTU"
}

legend_mapping = {
    "sb": r"[$\mathcal{R}_{ARH},\mathcal{R}_{SB}$]",
    "sb_sbs": r"[$\mathcal{R}_{ARH},\mathcal{R}_{SB}, \mathcal{F}_{SBF}$]",
    "sbs": r"[$\mathcal{R}_{ARH},\mathcal{F}_{SBF}$]",
    "abfta": r"[$\mathcal{R}_{ARH},\mathcal{F}_{ABFTU}$]",
    "sbs_abfta": r"[$\mathcal{R}_{ARH}, \mathcal{F}_{SBF}, \mathcal{F}_{ABFTU}$]"
}


heatmap_color = "viridis"
diagonal_color = "darkgrey"
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
    9: "90+"
}
phi0 = np.array([0.972, 0.992, 0.984, 0.987, 0.977, 0.971, 0.958, 0.926, 0.956, 0.926])
delta2_star = 0.756
hospitalization_risks = (1 - phi0) * delta2_star

MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/experiments/results/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/ref_results/seed_0/6obj_3days_crashed/model_9.pt"  # your single best model

action_colors = {
    "s": "hotpink",
    "w": "royalblue",
    "l": "dimgrey"
}
dpi = 500
def set_params_combined():
    plt.rcParams['legend.fontsize'] = 24  # 14   # legend text Trajectories
    plt.rcParams['axes.titlesize'] = 28  # 16   # axes & figure titles Trajectories
    plt.rcParams['axes.labelsize'] = 24  # 14   # x/y axis labels Trajectories
    plt.rcParams['xtick.labelsize'] = 20  # 12   # x-tick numbers Trajectories
    plt.rcParams['ytick.labelsize'] = 20  # 12   # y-tick numbers Trajectories


def set_params_3d():
    plt.rcParams['legend.fontsize'] = 18  # legend text Trajectories
    plt.rcParams['axes.titlesize'] = 26  # axes & figure titles Trajectories
    plt.rcParams['axes.labelsize'] = 20  # x/y axis labels Trajectories
    plt.rcParams['xtick.labelsize'] = 14  # x-tick numbers Trajectories
    plt.rcParams['ytick.labelsize'] = 14  # y-tick numbers Trajectories
    plt.rcParams['legend.loc'] = 'upper right'  # adjust as fallback
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.markerscale'] = 1.2
    plt.rcParams['legend.borderaxespad'] = 0.5
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['text.usetex'] = False  # if using LaTeX rendering, set to True
    plt.rcParams['font.family'] = 'serif'  # gives LaTeX-like look
    plt.rcParams['axes.facecolor'] = '#ffffff'  # lighter gray background
    plt.rcParams['figure.facecolor'] = '#ffffff'


def set_params_policies():
    plt.rcParams['legend.fontsize'] = 12  # legend text Trajectories
    plt.rcParams['axes.titlesize'] = 20  # axes & figure titles Trajectories
    plt.rcParams['axes.labelsize'] = 14  # x/y axis labels Trajectories
    plt.rcParams['xtick.labelsize'] = 14  # x-tick numbers Trajectories
    plt.rcParams['ytick.labelsize'] = 14  # y-tick numbers Trajectories
    plt.rcParams['legend.loc'] = 'upper right'  # adjust as fallback
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.markerscale'] = 1.2
    plt.rcParams['legend.borderaxespad'] = 0.5
    plt.rcParams['xtick.major.pad'] = 10
    plt.rcParams['text.usetex'] = False  # if using LaTeX rendering, set to True
    plt.rcParams['font.family'] = 'serif'  # gives LaTeX-like look
    plt.rcParams['axes.facecolor'] = '#ffffff'  # lighter gray background
    plt.rcParams['figure.facecolor'] = '#ffffff'


def interpolate_runs(runs, w=100):
    # Create a common x_grid from all runs’ first column values
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:, 0] for r in runs]))))
    all_values = np.stack([np.interp(all_steps, r[:, 0], r[:, 1]) for r in runs], axis=0)
    return all_steps, all_values


# Interpolate a list of Pareto-front arrays (columns: x, y, z) onto a common x grid (union of all x values)
def interpolate_runs_xyz(runs):
    """
    Interpolate a list of Pareto‑front arrays (columns: x, y, z) onto a
    common x grid (union of all x values). Returns
        x_vals, y_matrix, z_matrix
    where y_matrix and z_matrix have shape (n_runs, len(x_vals)).
    """
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:, 0] for r in runs]))))
    y_vals = np.stack([np.interp(all_steps, r[:, 0], r[:, 1]) for r in runs], axis=0)
    z_vals = np.stack([np.interp(all_steps, r[:, 0], r[:, 2]) for r in runs], axis=0)
    return all_steps, y_vals, z_vals


def load_runs_from_logdir(logdir, objectives):
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
            pareto_front = pareto_front[:, objectives]
            runs.append({'pareto_front': pareto_front})
    return runs


def plot_fixed_data(measure, objectives, ax=None, interactive=False, go_fig=None):
    """
    Plot or add to a Plotly figure the fixed‑policy reference points.

    Parameters
    ----------
    measure : str
        Name used to find ``fixed_policy_{measure}.csv``.
    objectives : list[int]
        Indexes of the objectives we want to visualise (length 2 or 3 expected).
    ax : matplotlib axis, optional
        If given, points are drawn on this axis (for static Matplotlib 2‑D/3‑D).
    interactive : bool, default False
        When True, we assume a Plotly context and ignore *ax*.
    go_fig : plotly.graph_objects.Figure, optional
        Pass an existing Figure to which we will add a trace.
    """
    csv_path = f"fixed_policy_{measure}.csv"
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found")
        return None

    df_fixed = pd.read_csv(csv_path)

    if measure == "abfta" or "sbs_abfta":
        df_fixed.drop(df_fixed.tail(1).index, inplace=True)  # drop last n rows

    # pick the first len(objectives) columns (or named o_0, o_1, …)
    if {'o_0', 'o_1'}.issubset(df_fixed.columns) and len(objectives) >= 2:
        x = df_fixed['o_0'].values
        y = df_fixed['o_1'].values
    else:
        x = df_fixed.iloc[:, 0].values
        y = df_fixed.iloc[:, 1].values

    if len(objectives) == 2:
        if interactive and go_fig is not None:
            go_fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers", name="fixed policy",
                    marker=dict(size=5, symbol="cross", color="black")
                )
            )
        elif ax is not None:
            ax.scatter(x, y, s=50, alpha=0.7, edgecolors='k',
                       label="fixed", marker='x', color="black")
        return x, y

    # --- 3‑D ----------------------------------------------------------------
    if len(objectives) == 3:
        if {'o_2'}.issubset(df_fixed.columns):
            z = df_fixed['o_2'].values
        else:
            z = df_fixed.iloc[:, 2].values

        if interactive and go_fig is not None:
            go_fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z, mode="markers", name="fixed policy",
                    marker=dict(size=4, symbol="cross", color="black")
                )
            )
        elif ax is not None:  # fallback static 3‑D scatter
            ax.scatter(x, y, z, s=30, alpha=0.7, edgecolors='k',
                       label="fixed", color="black", marker="x")
        return x, y, z


def plot_pareto_front_from_dir(measure, objectives, logdir, budget_label, scale,
                               save=False, use_interpolation=True, plot_individual=False):
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

    runs = load_runs_from_logdir(logdir, objectives)

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

    # ------------------------------------------------------------------ #
    # 3‑objective case: draw a 3‑D surface comparison instead of 2‑D      #
    # ------------------------------------------------------------------ #
    if len(objectives) == 3:
        # Use interpolation across seeds to obtain a common grid
        x_vals, y_vals, z_vals = interpolate_runs_xyz(sorted_runs)
        mean_y = np.mean(y_vals, axis=0)
        std_y = np.std(y_vals, axis=0)
        mean_z = np.mean(z_vals, axis=0)
        std_z = np.std(z_vals, axis=0)

        x = x_vals * scale[objectives[0]]
        y = mean_y * scale[objectives[1]]
        z = mean_z * scale[objectives[2]]
        return x, y, z

    else:
        # Interpolate across seeds on a common grid.
        x_vals, y_vals = interpolate_runs(sorted_runs)
        # Scale the x and y values.
        x_vals_scaled = x_vals * scale[objectives[0]]
        y_vals_scaled = y_vals * scale[objectives[1]]
        # Compute mean and standard deviation across runs.
        mean_curve = np.mean(y_vals_scaled, axis=0)
        std_curve = np.std(y_vals_scaled, axis=0)
        return x_vals_scaled, mean_curve, std_curve


def make_budget_plots(measure, scale, results_directory, repr_policies, save_dir=None, interactive=False):
    # The top-level folder containing budget_{i} subdirs
    plt.rcParams["figure.figsize"] = (17, 15)
    budget_symbol_map = {0: "o", 2: "s", 3: "D", 4: "^", 5: "v"}
    # Suppose you have budgets 0..5:
    all_budgets = [0, 2, 3, 4, 5]
    # We'll store the (x_vals, mean_curve, std_curve) in a dict
    results_dict = {}
    set_params_3d()
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

    # Step A: Generate one figure per budget
    for b in all_budgets:
        subdir = os.path.join(results_directory, f"budget_{b}")

        if len(objectives) == 2:
            x_vals_scaled, mean_curve, std_curve = plot_pareto_front_from_dir(
                measure,
                objectives,
                logdir=subdir,
                budget_label=str(b),
                scale=scale,
                save=False,
                plot_individual=False
            )
            if x_vals_scaled is not None:
                # Store for final combined plot
                results_dict[b] = (x_vals_scaled, mean_curve, std_curve)
        elif len(objectives) == 3:
            # Fetch the interpolated 3‑D Pareto front for this budget
            x_scaled, y_scaled, z_scaled = plot_pareto_front_from_dir(
                measure,
                objectives,
                logdir=subdir,
                budget_label=str(b),
                scale=scale,
                save=False,
                use_interpolation=True
            )
            if x_scaled is not None:
                results_dict[b] = (x_scaled, y_scaled, z_scaled)

    # === Combined 2-D plot for all budgets ===
    if len(objectives) == 2 and results_dict:
        plt.figure()
        for b, (x_vals_scaled, mean_curve, std_curve) in results_dict.items():
            b_label = "∞" if b == 0 else b
            plt.plot(
                x_vals_scaled,
                mean_curve,
                linewidth=3,
                label=f'Budget={b_label}',
                color=budget_colors[b]
            )
            plt.fill_between(
                x_vals_scaled,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.1
            )
            # --- mark representative policies for this budget ---
            if repr_policies:
                # pick out rows where the first element (budget) matches b
                repr_for_b = [r for r in repr_policies if int(r[0]) == b]
                for idx, r in enumerate(repr_for_b):
                    # r is [budget, ARI, ARH, SB_W, SB_S, SB_L, SB, SBS, ABFTA]
                    # extract and scale coordinates for current objectives
                    x_r = float(r[1 + objectives[0]]) * scale[objectives[0]]
                    y_r = float(r[1 + objectives[1]]) * scale[objectives[1]]
                    # plot with a distinct diamond marker
                    label = 'Representative' if idx == 0 else None
                    # plt.scatter(x_r, y_r,
                    #             s=100,
                    #             marker='D',
                    #             color=budget_colors.get(b, 'black'),
                    #             edgecolors='k',
                    #             label=label,
                    #             zorder=5)
                    symbol_r = budget_symbol_map.get(b, "D")
                    policy_color = r[-2] if len(r) > len(objectives) + 2 else budget_colors.get(b, "black")
                    line_dash = r[-3] if len(r) > len(objectives) + 3 else "dot"
                    if line_dash == "solid":
                        zorder = 6
                    else:
                        zorder = 5


                    # Scatter for the policy itself (policy-specific colour)
                    plt.scatter(x_r, y_r,
                                s=300,
                                marker=symbol_r,
                                color=policy_color,
                                edgecolors="k",
                                zorder=zorder)

                    # One invisible‑data scatter (only once per budget) to put a legend
                    # entry coloured like the budget’s line.
                    if idx == 0:
                        plt.scatter([], [],               # no data – just for legend
                                    s=300,
                                    marker=symbol_r,
                                    color=budget_colors.get(b, "black"),
                                    edgecolors="k",
                                    label="Representative")
        plot_fixed_data(measure, objectives, ax=plt.gca())
        plt.xlabel(f'{objectives_labels[objectives[0]]}')
        plt.ylabel(f'{objectives_labels[objectives[1]]}')
        plt.title(f'Pareto fronts for {objectives_labels[objectives[0]]} and '
                  f'{objectives_labels[objectives[1]]}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{measure}_results/budget_0/pf_budgets_{measure}.jpg", dpi=dpi)
        else:
            plt.show()

    # === Combined 3‑D surface for all budgets ============================
    if len(objectives) == 3 and results_dict:
        set_params_3d()
        if interactive and go is not None:
            fig = go.Figure()

            # Add Pareto points for each budget
            for b, (x_scaled, y_scaled, z_scaled) in results_dict.items():
                b_label = "∞" if b == 0 else b
                fig.add_trace(
                    go.Scatter3d(
                        x=x_scaled, y=y_scaled, z=z_scaled,
                        mode="markers",
                        marker=dict(size=3,
                                    color=budget_colors.get(b, "gray")),
                        name=f"Budget={b_label}"
                    )
                )

            # Add fixed‑policy reference
            plot_fixed_data(measure, objectives,
                            interactive=True, go_fig=fig)

            fig.update_layout(
                scene=dict(
                    xaxis_title=f'{objectives_labels[objectives[0]]}',
                    yaxis_title=f'{objectives_labels[objectives[1]]}',
                    zaxis_title=f'{objectives_labels[objectives[2]]}'
                ),
                title=f'Pareto fronts for {objectives_labels[objectives[0]]}, '
                      f'{objectives_labels[objectives[1]]} and {objectives_labels[objectives[2]]}'
            )
            fig.show()
        else:
            # Static Matplotlib 3-D plot with fixed orientation
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for b, (x_scaled, y_scaled, z_scaled) in results_dict.items():
                b_label = "∞" if b == 0 else b
                plt.plot(
                    x_scaled,
                    y_scaled,
                    z_scaled,
                    linewidth=3,
                    label=f'Budget={b_label}',
                    color=budget_colors[b]
                )
                if repr_policies:
                    # pick out rows where the first element (budget) matches b
                    repr_for_b = [r for r in repr_policies if int(r[0]) == b]
                    for idx, r in enumerate(repr_for_b):
                        # r is [budget, ARI, ARH, SB_W, SB_S, SB_L, SB, SBS, ABFTA]
                        # extract and scale coordinates for current objectives
                        x_r = float(r[1 + objectives[0]]) * scale[objectives[0]]
                        y_r = float(r[1 + objectives[1]]) * scale[objectives[1]]
                        z_r = float(r[1 + objectives[2]]) * scale[objectives[2]]
                        symbol_r = budget_symbol_map.get(b, "D")
                        policy_color = r[-2] if len(r) > len(objectives) + 2 else budget_colors.get(b, "black")
                        line_dash = r[-3] if len(r) > len(objectives) + 3 else "dot"
                        if line_dash == "solid":
                            zorder = 6
                        else:
                            zorder = 5
                        # Scatter for the policy itself
                        ax.scatter(x_r, y_r, z_r,
                                   s=200,
                                   marker=symbol_r,
                                   color=policy_color,
                                   edgecolors='k',
                                   zorder=zorder)

                        # Legend proxy (once per budget)
                        if idx == 0:
                            ax.scatter([], [], [],             # empty data
                                       s=200,
                                       marker=symbol_r,
                                       color=budget_colors.get(b, "black"),
                                       edgecolors='k',
                                       label="Representative",
                                       zorder=zorder)
            # Overlay fixed-policy reference
            plot_fixed_data(measure, objectives, ax=ax)
            # Format tick labels with scientific notation
            from matplotlib.ticker import ScalarFormatter
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                axis.set_major_formatter(fmt)

            # Axis labels with scale factor annotation
            ax.set_xlabel(f"{objectives_labels[objectives[0]]}", labelpad=20)
            ax.set_ylabel(f"{objectives_labels[objectives[1]]}", labelpad=20)
            # set z-axis label with extra padding and rotate so it sits to the left
            ax.set_zlabel(f"{objectives_labels[objectives[2]]}", labelpad=30)
            ax.zaxis.label.set_rotation(90)
            ax.zaxis.label.set_horizontalalignment('left')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # white background
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.set_title(f'Pareto fronts for {objectives_labels[objectives[0]]}, '
                         f'{objectives_labels[objectives[1]]} and {objectives_labels[objectives[2]]} ', y=1.02)
            # Set custom view
            if measure == "sbs_abfta":
                ax.legend(loc='upper right', bbox_to_anchor=(0.96, 0.93))
                fig.subplots_adjust(left=-0.03, right=0.95, top=0.95, bottom=-0.01)
                ax.view_init(elev=20, azim=-60)
            else:
                ax.legend(loc='upper right', bbox_to_anchor=(0.96, 0.93))
                fig.subplots_adjust(left=-0.03, right=0.95, top=0.95, bottom=-0.01)
                ax.view_init(elev=20, azim=-60)
            # --- Apply tight_layout and reduce whitespace before saving/showing ---
            if save_dir:
                plt.tight_layout()
                #ax.set_position([0.01, 0.02, 0.96, 0.96])
                plt.savefig(f"{save_dir}/{measure}_results/budget_0/pf_budgets_{measure}.jpg", dpi=dpi)
            else:
                plt.tight_layout()
                #ax.set_position([0.01, 0.02, 0.96, 0.96])
                plt.show()


device = 'cpu'


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

def _load_last_lost_matrix(data_dir, pid):
    """
    Load the final lost_contacts matrix for a given policy id (or -1 for last).
    Returns a 10x10 numpy array or None if missing.
    """
    import os, json
    import numpy as np
    import pandas as pd
    if pid == -1:
        subs = [d for d in os.listdir(data_dir)
                if d.isdigit() and os.path.isdir(os.path.join(data_dir, d))]
        if not subs:
            return None
        subdir = str(max(int(d) for d in subs))
    else:
        subdir = str(pid)
    csv_path = os.path.join(data_dir, subdir, 'run_0.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, parse_dates=['dates'],
                     converters={'lost_matrices': json.loads})
    if 'lost_matrices' not in df.columns:
        return None
    mat = np.abs(np.array(df['lost_matrices'].iloc[-1]))
    return mat


def compare_two_measures_heatmaps(
        measure1: str,
        measure2: str,
        model_paths_dir: str,
        budget: int,
        goal_hospitalizations_list: list[float],
        save_dir: str | None = None,
        pid1_list: list[int] | None = None,
        pid2_list: list[int] | None = None,
        tol: float = 600
):
    """
    For each hospitalization target in *goal_hospitalizations_list* pick the
    policy from every measure whose **total** hospitalizations is within ±tol of
    the target (unless an explicit pid list is supplied).  Plot the final
    lost‑contacts matrix for measure1 vs measure2 as side‑by‑side heat‑maps.

    Signature mirrors `compare_two_measures_multiple_trajectories` so the same
    call‑sites can reuse it.
    """
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    set_params_policies()

    include_heat = ("abfta" in (measure1, measure2)) or ("sbs_abfta" in (measure1, measure2))
    if not include_heat:
        return  # nothing to do

    # ------------------------------------------------------------------ #
    # helper for policy selection                                         #
    # ------------------------------------------------------------------ #
    def _pick_pid(tdir: str, pid_hint: int | None, target: float):
        if pid_hint is not None:
            if pid_hint == -1:
                nums = [int(d) for d in os.listdir(tdir) if d.isdigit()]
                return max(nums) if nums else None
            return pid_hint
        return select_policies_by_total_hospitalizations(tdir, target, tol)

    pid1_list = pid1_list or [None] * len(goal_hospitalizations_list)
    pid2_list = pid2_list or [None] * len(goal_hospitalizations_list)

    tdir1 = os.path.join(model_paths_dir,
                         f"{measure1}_results", f"budget_{budget}", "policies-transitions")
    tdir2 = os.path.join(model_paths_dir,
                         f"{measure2}_results", f"budget_{budget}", "policies-transitions")

    n_rows = len(goal_hospitalizations_list)
    # Slightly narrower width so the two heat‑maps sit closer;
    # more height per row for titles/colour‑bar.
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(12, 4.8 * n_rows),
                             squeeze=False)
    # reduce horizontal spacing between subplots
    fig.subplots_adjust(wspace=0.06, hspace=0.35)

    age_labels = ['0-10','10-20','20-30','30-40','40-50',
                  '50-60','60-70','70-80','80-90','90+']

    for r, (target, pid1_hint, pid2_hint) in enumerate(zip(goal_hospitalizations_list,
                                                           pid1_list, pid2_list)):
        pid1 = _pick_pid(tdir1, pid1_hint, target)
        pid2 = _pick_pid(tdir2, pid2_hint, target)

        mat1 = _load_last_lost_matrix(tdir1, pid1)
        mat2 = _load_last_lost_matrix(tdir2, pid2)

        for c, (mat, meas, pid) in enumerate(((mat1, measure1, pid1), (mat2, measure2, pid2))):
            ax = axes[r, c]
            if mat is not None:
                im = ax.imshow(mat, cmap=heatmap_color, origin='upper')
                # shade the diagonal cells
                for i in range(len(age_labels)):
                    rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                             linewidth=0,
                                             edgecolor=None,
                                             facecolor=diagonal_color,
                                             zorder=3)
                    ax.add_patch(rect)
            formatted_target = format(target, ",")
            ax.set_title(f"{measure_mapping[meas]} ({formatted_target} Hospitalizations)", fontsize=12)
            ax.set_xticks(range(len(age_labels)))
            ax.set_xticklabels(age_labels, rotation=45, ha='right')
            ax.tick_params(axis='x', pad=2)
            if c == 0:
                ax.set_yticks(range(len(age_labels)))
                ax.set_yticklabels(age_labels)
            else:
                ax.set_yticks([])

        # colour‑bar for this row next to right heat‑map
        cbar = fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel("Proportionally Lost contacts", rotation=-90, va='bottom', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.tick_params(axis='x', pad=8)
        # adjust title font size for readability
        axes[r, 0].title.set_fontsize(12)
        axes[r, 1].title.set_fontsize(12)

    fig.suptitle(f"Proportionally Lost Intergroup Contacts",
                 fontsize=16, y=0.99)
    if save_dir:
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(
            save_dir,
            f"heatmaps_{measure1}+{measure2}_goals.jpg"
        )
        fig.savefig(fname, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def compare_measures_heatmaps_by_pids(
        measure1: str,
        measure2: str,
        model_paths_dir: str,
        budget: int,
        pid1_list: list[int],
        pid2_list: list[int],
        save_dir: str | None = None
):
    """
    Heat‑map counterpart of `compare_measures_multiple_trajectories_by_pids`.
    Produces one row per (pid1, pid2) pair: left heat‑map = measure1, right =
    measure2.
    """
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    set_params_policies()

    include_heat = ("abfta" in (measure1, measure2)) or ("sbs_abfta" in (measure1, measure2))
    if not include_heat:
        return

    if len(pid1_list) != len(pid2_list):
        raise ValueError("pid1_list and pid2_list must have the same length")

    def _resolve_pid(tdir: str, hint: int):
        if hint != -1:
            return hint
        nums = [int(d) for d in os.listdir(tdir) if d.isdigit()]
        return max(nums) if nums else None

    tdir1 = os.path.join(model_paths_dir,
                         f"{measure1}_results", f"budget_{budget}", "policies-transitions")
    tdir2 = os.path.join(model_paths_dir,
                         f"{measure2}_results", f"budget_{budget}", "policies-transitions")

    n_rows = len(pid1_list)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(12, 4.8 * n_rows),
                             squeeze=False)
    fig.subplots_adjust(wspace=0.06, hspace=0.35)

    age_labels = ['0-10','10-20','20-30','30-40','40-50',
                  '50-60','60-70','70-80','80-90','90+']
    print(f' {measure1} - {measure2}')
    for r, (pid1_hint, pid2_hint) in enumerate(zip(pid1_list, pid2_list)):
        pid1 = _resolve_pid(tdir1, pid1_hint)
        pid2 = _resolve_pid(tdir2, pid2_hint)

        mat1 = _load_last_lost_matrix(tdir1, pid1)
        mat2 = _load_last_lost_matrix(tdir2, pid2)

        for c, (mat, meas, pid) in enumerate(((mat1, measure1, pid1), (mat2, measure2, pid2))):
            ax = axes[r, c]
            if mat is not None:
                im = ax.imshow(mat, cmap=heatmap_color, origin='upper')
                # shade the diagonal cells
                for i in range(len(age_labels)):
                    rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                             linewidth=0,
                                             edgecolor=None,
                                             facecolor=diagonal_color,
                                             zorder=3)
                    ax.add_patch(rect)
            ax.set_title(f"{measure_mapping[meas]} (policy {pid})", fontsize=12)
            ax.set_xticks(range(len(age_labels)))
            ax.set_xticklabels(age_labels, rotation=45, ha='right')
            ax.tick_params(axis='x', pad=2)
            if c == 0:
                ax.set_yticks(range(len(age_labels)))
                ax.set_yticklabels(age_labels)
            else:
                ax.set_yticks([])

        cbar = fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel("Proportionally Lost contacts", rotation=-90, va='bottom', fontsize=10)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.tick_params(axis='x', pad=8)
        # adjust title font size for readability
        axes[r, 0].title.set_fontsize(12)
        axes[r, 1].title.set_fontsize(12)

    fig.suptitle(f"Proportionally Lost Intergroup Contacts",
                 fontsize=16, y=0.99)
    if save_dir:
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(
            save_dir,
            f"heatmaps_{measure1}+{measure2}_pids.jpg"
        )
        fig.savefig(fname, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)

def save_lost_matrices_heatmaps(data_dir, policy_ids, save_dir, label_dir):
    """
    For each policy in policy_ids, re-load its run_0.csv from data_dir,
    grab the final 'lost_matrices' entry, and save it as a 300 dpi heatmap.

    Parameters
    ----------
    data_dir : str
        Base folder where each policy lives in a subdirectory named "0", "1", …
    policy_ids : list[int]
        List of policy IDs to render (use -1 to mean “the highest numbered folder”).
    save_dir : str or None
        Where to write the JPEGs; if None, each figure is plt.show()n instead.
    label_dir : str
        Prefix for the output filenames, e.g. the same `label_dir` you pass to plot_policies.
    """
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    set_params_policies()

    # prepare subplots: one column per policy
    n = len(policy_ids)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)
    # When constrained_layout=True, do not call subplots_adjust (avoids warning)
    # ensure axes is iterable
    if n == 1:
        axes = [axes]

    age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50',
                  '50-60', '60-70', '70-80', '80-90', '90+']

    # collect the last image handle for the colorbar
    im = None
    for ax, pid in zip(axes, policy_ids):
        # determine subdir for this pid
        if pid == -1:
            subs = [d for d in os.listdir(data_dir)
                    if d.isdigit() and os.path.isdir(os.path.join(data_dir, d))]
            if not subs:
                continue
            subdir = str(max(int(d) for d in subs))
        else:
            subdir = str(pid)

        csv_path = os.path.join(data_dir, subdir, 'run_0.csv')
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, parse_dates=['dates'],
                         converters={'lost_matrices': json.loads})
        if 'lost_matrices' not in df.columns:
            continue

        mat = np.abs(np.array(df['lost_matrices'].iloc[-1]))

        # plot on this axis
        im = ax.imshow(mat, cmap=heatmap_color, origin='upper')
        ax.set_title(f'Policy {subdir}', pad=10, fontsize=16)
        # Always show y-axis ticks and labels
        ax.set_xticks(range(len(age_labels)))
        ax.set_xticklabels(age_labels, rotation=45)
        # Only show x-axis ticks on the first subplot
        if ax is axes[0]:
            ax.set_yticks(range(len(age_labels)))
            ax.set_yticklabels(age_labels, ha='right')
        else:
            ax.set_yticks([])
        ax.tick_params(pad=3)

        for i in range(len(age_labels)):
            rect = patches.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                     linewidth=0,
                                     edgecolor=None,
                                     facecolor=diagonal_color,
                                     zorder=3)
            ax.add_patch(rect)

    # add shared colorbar
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
        cbar.ax.set_ylabel('Proportionally Lost contacts', rotation=-90, va='bottom')

    # super-title and save
    fig.suptitle(f'Proportionally Lost Intergroup Contacts', fontsize=20)
    fname = f"{label_dir}_lost_matrices.jpg"
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, fname), dpi=dpi)
    else:
        plt.show()
    plt.close(fig)

def plot_policies(policy_ids, objectives, data_dir, label_dir, save_dir):
    figsize_x = 13
    figsize_y = 3 * len(policy_ids)
    figsize = (figsize_x, figsize_y)

    fig, axes = plt.subplots(nrows=len(policy_ids), ncols=1, sharex=True, figsize=figsize)
    set_params_policies()
    date_fmt = DateFormatter("%d/%m")
    title_policies = ""

    for ax, pid in zip(axes, policy_ids):
        set_params_policies()
        # find the run CSV inside the numbered policy folder
        if pid == -1:
            subdirs = [
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
            ]
            if not subdirs:
                raise FileNotFoundError(f"No numeric policy subdirectories in {data_dir}")
            last = str(max(int(d) for d in subdirs))
            title_policies += last
            print(f"Using last policy folder: {last}")
            policy_folder = os.path.join(data_dir, last)
        else:
            print(pid)
            title_policies += str(pid)
            policy_folder = os.path.join(data_dir, str(pid))
        title_policies += ", "
        run_files = glob.glob(os.path.join(policy_folder, 'run*.csv'))
        if not run_files:
            raise FileNotFoundError(f"No run CSV file found in {policy_folder}")
        csv_path = run_files[0]
        df = pd.read_csv(csv_path, parse_dates=['dates'])

        # Add vertical lines for school holidays: July 1 to September 1
        #holiday_start = pd.to_datetime("2020-07-01")
        #ax.axvline(holiday_start, color='darkgrey', linestyle='--', linewidth=2)

        # left axis
        ax.plot(df['dates'], df['i_hosp_new'], color='deepskyblue', label='daily new hosp')
        ax.plot(df['dates'], df['i_icu_new'], color='orange', label='daily new ICU')
        ax.plot(df['dates'], df['d_new'], color='red', label='daily deaths')
        ax.set_ylabel('individuals', fontsize=14)
        ax.yaxis.label.set_fontfamily('serif')
        ax.set_ylim(0, 4500)
        ax.tick_params(axis='y', labelcolor='black')
        set_params_policies()

        # right axis
        ax2 = ax.twinx()
        ax2.plot(df['dates'], df['p_w'], '--', color=action_colors["w"], label='$p_w$')
        ax2.plot(df['dates'], df['p_s'], '--', color=action_colors["s"], label='$p_s$')
        ax2.plot(df['dates'], df['p_l'], '--', color=action_colors["l"], label='$p_l$')
        labelpad = 35 if ax == axes[0] else 5
        ax2.set_ylabel('proportion', labelpad=labelpad)
        ax2.yaxis.label.set_fontfamily('serif')
        ax2.set_ylim(-0.05, 1.05)

        # Apply consistent serif styling to both left and right y-axes
        for side in [ax, ax2]:
            side.tick_params(axis='y', labelsize=14, labelcolor='black')
            for label in side.get_yticklabels():
                label.set_fontfamily('serif')

        # Also set font family for bottom x-axis and right y-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontfamily('serif')
        for label in ax2.get_yticklabels():
            label.set_fontfamily('serif')

        # Hide right tick labels only on top plot (axes[0]), not middle or bottom
        if ax == axes[0]:
            ax2.set_yticklabels([])
        # formatting
        ax.xaxis.set_visible(False)  # hide all x‐ticks until the bottom plot
        ax2.xaxis.set_visible(False)

    # bottom plot gets the time axis
    axes[-1].xaxis.set_visible(True)
    axes[-1].set_xlabel('time', fontsize=14)
    axes[-1].xaxis.label.set_fontfamily('serif')
    axes[-1].xaxis.set_major_formatter(date_fmt)
    fig.autofmt_xdate()
    # Strip trailing comma and space from title_policies
    title_policies = title_policies.rstrip(', ')
    # single legend at top
    # collect all handles & labels from the top subplot
    handles = []
    labels = []
    for axis in (axes[0], axes[0].twinx()):
        h, l = axis.get_legend_handles_labels()
        handles += h
        labels += l
    if len(objectives) == 2:
        fig.suptitle(f"Policy trajectories {title_policies} for {objectives_labels[objectives[0]]} and "
                     f"{objectives_labels[objectives[1]]} ",
                     fontsize=18, y=0.97)
    else:
        fig.suptitle(f"Policy trajectories {title_policies} for {objectives_labels[objectives[0]]}, "
                     f"{objectives_labels[objectives[1]]} and {objectives_labels[objectives[2]]} ",
                     fontsize=18, y=0.97)  # slightly lower to fit under legend

    # Then the legend just above the first plot
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=6, frameon=False)

    # Legend for action probabilities below main legend
    from matplotlib.lines import Line2D
    action_handles = [
        Line2D([0], [0], color=action_colors["w"], linestyle='--'),
        Line2D([0], [0], color=action_colors["s"], linestyle='--'),
        Line2D([0], [0], color=action_colors["l"], linestyle='--')
    ]
    action_labels = ['work', 'school', 'leisure']
    fig.legend(action_handles, action_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.92),
               ncol=3, frameon=False)

    # Set date tick label size and font on bottom axis
    for label in axes[-1].get_xticklabels():
        label.set_fontsize(14)
        label.set_fontfamily('serif')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # make room for the legend
    if save_dir:
        plt.savefig(f"{save_dir}/{label_dir}_trajectories.jpg", dpi=dpi)
    else:
        plt.show()

    if label_dir == "abfta" or label_dir == "sbs_abfta":
        save_lost_matrices_heatmaps(data_dir, policy_ids, save_dir, label_dir)
    # Also plot lost contacts per age group for each policy
    # plot_lost_contacts_per_age(policy_ids, data_dir)


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
    ax2.plot(df['dates'], df['p_w'], '--', color=action_colors["w"], label='$p_w$')
    ax2.plot(df['dates'], df['p_s'], '--', color=action_colors["s"], label='$p_s$')
    ax2.plot(df['dates'], df['p_l'], '--', color=action_colors["l"], label='$p_l$')
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
            selected.append(int(d))
    return min(selected, key=lambda x: abs(x - target))


def select_policy(transitions_dir, measure, obj_col):
    pid = None
    best = -np.inf
    for d in os.listdir(transitions_dir):
        if not d.isdigit():
            continue
        csv_path = os.path.join(transitions_dir, d, "run_0.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, parse_dates=["dates"])
        if obj_col not in df.columns:
            raise KeyError(f"Expected column '{obj_col}' in {csv_path}")
        total_obj = df[obj_col].sum()
        if total_obj > best:
            pid = int(d)
    return pid


def main(measure, goal_hospitalizations, seed, save_dir=None, run_episodes=False):
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
                                                                     tol=600)
        #policy_objective = select_policy(transitions_dir, measure, "o_6")
        
        idx = [inspect_policies]
        idx = [0, inspect_policies, -1]
        # plot_multiple_policies_with_contacts(idx, transitions_dir, measure)
        plot_policies(idx, objectives, transitions_dir, measure, save_dir)
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


def plot_two_age_groups_comparison(goal_hospitalizations_list, measures, model_paths_dir, budget, age_indices, legend_pos, tol=600, save_dir=None):
    """
    For each target in goal_hospitalizations_list:
      - select the policy for each measure whose total hospitalizations is closest to the target (±tol),
      - plot lost contacts over time for two specified age groups in side-by-side subplots,
      - combine all targets into one figure with one row per target,
      - show dates only on the bottom row,
      - add a subtitle per row with the target hospitalizations,
      - and save or show the combined figure.
    """
    import os
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    set_params_policies()
    date_fmt = DateFormatter("%d/%m")

    age1, age2 = age_indices
    age_labels = [age_groups[age1], age_groups[age2]]

    n = len(goal_hospitalizations_list)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), sharey=True)
    # ensure axes is 2D array even if n == 1
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, target in enumerate(goal_hospitalizations_list):
        # select policy ids for this target
        pids = []
        for measure in measures:
            tdir = os.path.join(model_paths_dir, f"{measure}_results", f"budget_{budget}", "policies-transitions")
            pid = select_policies_by_total_hospitalizations(tdir, target, tol)
            pids.append(pid)

        # plot each of the two age groups
        for j, age_idx in enumerate(age_indices):
            ax = axes[i, j]
            for pid, measure in zip(pids, measures):
                base = os.path.join(model_paths_dir, f"{measure}_results", f"budget_{budget}", "policies-transitions")
                subdirs = [d for d in os.listdir(base) if d.isdigit()]
                if not subdirs:
                    continue
                folder = str(max(map(int, subdirs))) if pid == -1 else str(pid)
                csv_path = os.path.join(base, folder, 'run_0.csv')
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path, parse_dates=['dates'], converters={'lost_contacts': json.loads})
                lost_arr = np.abs(np.vstack(df['lost_contacts'].tolist()))
                dates = df['dates']
                ax.plot(dates, lost_arr[:, age_idx], label=measure, linewidth=2)

                ax.grid(True, zorder=0)
                ax.tick_params(axis='x', pad=2)
                if ax is axes[1]:
                    ax.legend(loc=legend_pos, labels=legend_mapping.values())

            ax.set_title(f"Age group {age_labels[j]} ({target:,} Hospitalizations)", fontsize=12)
            ax.set_ylim(0, 0.8)
            ax.yaxis.set_ticks_position("left")
            ax.yaxis.set_label_position("left")
            ax.grid(True, zorder=0)

            if i == 0 and j == 0:
                ax.legend(loc=legend_pos, labels=legend_mapping.values())

            # show x-axis ticks only on bottom row
            if i == n - 1:
                ax.xaxis.set_major_formatter(date_fmt)
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
            else:
                ax.set_xticklabels([])

            # label shared y-axis on left column
            if j == 0:
                ax.set_ylabel("Proportionally Lost contacts")

    fig.suptitle("Proportionally Lost Contacts increasing Hospitalization goals", fontsize=20, y=0.99)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"lost_contacts_{age_groups[age_indices[0]]}_{age_groups[age_indices[1]]}.jpg")
        fig.savefig(fname, dpi=dpi)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_age_group_grid_comparison(goal_hospitalizations_list, measures, model_paths_dir, budget, tol=600, save_dir=None):
    """
    For each target in goal_hospitalizations_list:
      - select one policy per measure whose total hospitalizations is closest to the target (±tol),
      - plot lost contacts over time for each age group in a 5×2 grid,
      - and save one image per target.
    """
    import os
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import numpy as np
    set_params_combined()
    # iterate over each hospitalization target
    for goal in goal_hospitalizations_list:
        # select policy IDs for each measure
        pids = []
        for measure in measures:
            tdir = os.path.join(model_paths_dir,
                                 f"{measure}_results", f"budget_{budget}", "policies-transitions")
            pid = select_policies_by_total_hospitalizations(tdir, goal, tol)
            pids.append(pid)

        # set styling
        set_params_combined()
        right_ids = [0, 6, 7, 8, 9]
        left_ids = [1, 2, 3, 4, 5]
        age_cols = [left_ids, right_ids]

        fig, axes = plt.subplots(5, 2, figsize=(40, 40), sharex=True)
        date_fmt = DateFormatter("%d/%m")

        # plot each age group
        for col, age_list in enumerate(age_cols):
            for row, age_idx in enumerate(age_list):
                ax = axes[row, col]
                for pid, measure in zip(pids, measures):
                    base = os.path.join(model_paths_dir,
                                        f"{measure}_results",
                                        f"budget_{budget}",
                                        "policies-transitions")
                    # handle most recent run if pid == -1
                    subdirs = [d for d in os.listdir(base) if d.isdigit()]
                    folder = str(max(map(int, subdirs))) if pid == -1 else str(pid)
                    csv_path = os.path.join(base, folder, 'run_0.csv')
                    if not os.path.exists(csv_path):
                        continue
                    df = pd.read_csv(csv_path, parse_dates=['dates'], converters={'lost_contacts': json.loads})
                    lost_arr = np.abs(np.vstack(df['lost_contacts'].tolist()))
                    dates = df['dates']
                    ax.plot(dates, lost_arr[:, age_idx], label=measure, linewidth=3)

                ax.set_ylabel(age_groups[age_idx])
                ax.xaxis.set_major_formatter(date_fmt)
                ax.grid(True, zorder=0)
                ax.set_ylim(0, 0.8)

        # shared x-axis label
        for col in (0, 1):
            axes[-1, col].set_xlabel("Date")

        # legend
        fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                   ncol=len(measures), labels=legend_mapping.values())

        # title with extra top margin
        fig.suptitle("Proportionally Lost Contacts for different PCN versions", fontsize=28, y=0.995)

        # format date labels
        fig.autofmt_xdate()

        # adjust layout to leave room for title
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"lost_contacts_grid_goal_{goal}.jpg")
            fig.savefig(out_path, dpi=dpi)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


# ---------------------------------------------------------------------------
# New function: compare_two_measures_trajectories
# ---------------------------------------------------------------------------
def compare_two_measures_trajectories(measure1, measure2, model_paths_dir, budget, goal_hospitalizations,
                                      pid1=None, pid2=None, tol=300):
    """
    Compare epidemic trajectories for two measures side by side.

    Parameters
    ----------
    measure1, measure2 : str
        Names of the two measures (e.g. "sb", "sbs").
    model_paths_dir : str
        Base directory containing "<measure>_results" subfolders.
    budget : int
        Budget number used in folder names.
    goal_hospitalizations : float
        Target total hospitalizations for policy selection.
    tol : float, optional
        Tolerance for selecting policies based on total hospitalizations.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    # Build transitions directories
    dir1 = os.path.join(model_paths_dir, f"{measure1}_results", f"budget_{budget}", "policies-transitions")
    dir2 = os.path.join(model_paths_dir, f"{measure2}_results", f"budget_{budget}", "policies-transitions")

    # Select policy IDs
    if pid1 == None and pid2 == None:
        pid1 = select_policies_by_total_hospitalizations(dir1, goal_hospitalizations, tol)
        pid2 = select_policies_by_total_hospitalizations(dir2, goal_hospitalizations, tol)

    subdirs1 = [
        d for d in os.listdir(dir1)
        if os.path.isdir(os.path.join(dir1, d)) and d.isdigit()
    ]
    if pid1 == -1:
        pid1 = max(int(d) for d in subdirs1)

    subdirs2 = [
        d for d in os.listdir(dir2)
        if os.path.isdir(os.path.join(dir2, d)) and d.isdigit()
    ]
    if pid2 == -1:
        pid2 = max(int(d) for d in subdirs2)

    # Load the first run CSV for each
    path1 = os.path.join(dir1, str(pid1), "run_0.csv")
    path2 = os.path.join(dir2, str(pid2), "run_0.csv")
    df1 = pd.read_csv(path1, parse_dates=['dates'])
    df2 = pd.read_csv(path2, parse_dates=['dates'])

    # Prepare plotting
    fig, axes = plt.subplots(1, 2, figsize=(28, 5), sharey=True)
    date_fmt = DateFormatter("%d/%m")

    for ax, df, meas, pid in zip(axes, [df1, df2], [measure1, measure2], [pid1, pid2]):
        # Plot hospitalizations, ICU, deaths
        ax.plot(df['dates'], df['i_hosp_new'], color='deepskyblue', label='daily new hosp')
        ax.plot(df['dates'], df['i_icu_new'], color='orange', label='daily new ICU')
        ax.plot(df['dates'], df['d_new'], color='red', label='daily deaths')
        ax.set_title(f"{meas} (policy {pid})")
        ax.set_xlabel('Date')
        if ax is axes[0]:
            ax.set_ylabel('Individuals')
        ax.xaxis.set_major_formatter(date_fmt)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 4500)

        # Plot action probabilities on twin axis
        ax2 = ax.twinx()
        ax2.plot(df['dates'], df['p_w'], '--', color=action_colors["w"], label='$p_w$')
        ax2.plot(df['dates'], df['p_s'], '--', color=action_colors["s"], label='$p_s$')
        ax2.plot(df['dates'], df['p_l'], '--', color=action_colors["l"], label='$p_l$')
        ax2.set_ylim(-0.05, 1.05)

        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', ncol=2, frameon=False)

    plt.tight_layout()
    plt.show()


# Example:
# plot_age_group_grid_comparison(pids, all_measures, model_paths_dir, budget)


def plot_hospitalization_risk_bar(age_groups_dict, risk_array, save_dir=None):
    """
    Plot a bar chart of hospitalization risk per age group.

    :param age_groups_dict: dict mapping age_index to label (e.g. {0: '0-10', ...})
    :param risk_array: 1D array‐like of risk values (must align with the keys/order of age_groups_dict)
    """
    set_params_policies()
    import matplotlib.pyplot as plt

    # 1) Pick a light grid style
    # plt.style.use('ggplot')

    # 2) Prepare data
    indices = sorted(age_groups_dict.keys())
    labels = [age_groups_dict[i] for i in indices]
    values = [risk_array[i] for i in indices]

    # 3) Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # 4) Draw bars with custom color and edge
    bars = ax.bar(labels, values,
                  color='skyblue',
                  edgecolor='gray',
                  linewidth=0.8)
    ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.7, zorder=0)
    for bar in bars:
        bar.set_zorder(3)
    # 5) Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 6) Add value labels above each bar
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9)

    # 7) Axis labels & title styling
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Hospitalization Risk', fontsize=12)
    ax.set_title('Hospitalization Risk by Age Group',
                 fontsize=16,
                 fontweight='normal',  # remove bold
                 loc='center',
                 pad=20)  # optional: use serif for elegant style

    # 8) Tidy up the x-axis ticks
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/hospitalization_risks.jpg", dpi=dpi)
    else:
        plt.show()


def compare_two_measures_multiple_trajectories(
        measure1: str,
        measure2: str,
        model_paths_dir: str,
        budget: int,
        goal_hospitalizations_list: list[float],
        save_dir: str | None = None,
        pid1_list: list[int] | None = None,
        pid2_list: list[int] | None = None,
        tol: float = 600,
):
    """
    Plot trajectories for *measure1* vs *measure2* for a series of
    hospitalization targets, one row per target (left = measure1,
    right = measure2).

    All per‑panel legends are replaced by a **single** figure‑level legend
    positioned just under the suptitle (similar to `plot_policies`).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    set_params_policies()

    # Helper to pick a policy ID
    def _pick_policy(trans_dir: str, explicit_pid: int | None, target: float):
        if explicit_pid is not None:
            if explicit_pid == -1:                      # most‑recent run
                nums = [int(d) for d in os.listdir(trans_dir) if d.isdigit()]
                if not nums:
                    raise FileNotFoundError(f"No numeric subdirs in {trans_dir}")
                return max(nums)
            return explicit_pid
        # Auto‑select closest policy (by total hosp.)
        return select_policies_by_total_hospitalizations(trans_dir, target, tol)

    n_rows = len(goal_hospitalizations_list)
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(28, fig_height),
                             sharey=True, squeeze=False)
    date_fmt = DateFormatter("%d/%m")

    # Leave space for legend (suptitle a bit lower)
    fig.suptitle("Comparison of trajectories for different hospitalization goals",
                 fontsize=28, y=0.97)

    # Prepare optional pid lists
    pid1_list = pid1_list or [None] * n_rows
    pid2_list = pid2_list or [None] * n_rows
    if len(pid1_list) != n_rows or len(pid2_list) != n_rows:
        raise ValueError("pid*_list must match goal_hospitalizations_list length")

    # Disk paths (constant across rows)
    tdir1 = os.path.join(model_paths_dir,
                         f"{measure1}_results", f"budget_{budget}", "policies-transitions")
    tdir2 = os.path.join(model_paths_dir,
                         f"{measure2}_results", f"budget_{budget}", "policies-transitions")

    # --- storage for a single legend ------------------------------------
    legend_handles: list = []
    legend_labels: list = []

    for r, (target, pid1_hint, pid2_hint) in enumerate(zip(goal_hospitalizations_list,
                                                           pid1_list, pid2_list)):
        pid1 = _pick_policy(tdir1, pid1_hint, target)
        pid2 = _pick_policy(tdir2, pid2_hint, target)

        def _load_csv(tdir: str, pid: int) -> pd.DataFrame:
            return pd.read_csv(os.path.join(tdir, str(pid), "run_0.csv"),
                               parse_dates=["dates"])

        df1, df2 = _load_csv(tdir1, pid1), _load_csv(tdir2, pid2)
        ax_l, ax_r = axes[r]

        # ------ left panel ----------------------------------------------
        ax_l.plot(df1.dates, df1.i_hosp_new, color="deepskyblue", label="daily new hosp")
        ax_l.plot(df1.dates, df1.i_icu_new, color="orange",     label="daily new ICU")
        ax_l.plot(df1.dates, df1.d_new,       color="red",       label="daily deaths")
        ax_l.set_title(f"{target:,} Hospitalizations for {measure_mapping[measure1]}",
                       fontweight="normal")
        if r == n_rows - 1:
            ax_l.set_xlabel("Date")
        ax_l.set_ylabel("Individuals")
        ax_l.set_ylim(0, 4500)
        ax_l.xaxis.set_major_formatter(date_fmt)
        ax_l.tick_params(axis="x", rotation=45)

        ax2l = ax_l.twinx()
        ax2l.plot(df1.dates, df1.p_w, "--", color=action_colors["w"],    label="$p_w$")
        ax2l.plot(df1.dates, df1.p_s, "--", color=action_colors["s"],  label="$p_s$")
        ax2l.plot(df1.dates, df1.p_l, "--", color=action_colors["l"],   label="$p_l$")
        ax2l.set_ylim(-0.05, 1.05)

        # Collect legend items once (first row, left panel only)
        if r == 0:
            h1, l1 = ax_l.get_legend_handles_labels()
            h2, l2 = ax2l.get_legend_handles_labels()
            legend_handles = h1 + h2
            legend_labels = l1 + l2

        # ------ right panel ---------------------------------------------
        ax_r.plot(df2.dates, df2.i_hosp_new, color="deepskyblue", label="daily new hosp")
        ax_r.plot(df2.dates, df2.i_icu_new, color="orange",     label="daily new ICU")
        ax_r.plot(df2.dates, df2.d_new,       color="red",       label="daily deaths")
        ax_r.set_title(f"{target:,} Hospitalizations for {measure_mapping[measure2]}",
                       fontweight="normal")
        if r == n_rows - 1:
            ax_r.set_xlabel("Date")
        ax_r.xaxis.set_major_formatter(date_fmt)
        ax_r.tick_params(axis="x", rotation=45)

        ax2r = ax_r.twinx()
        ax2r.plot(df2.dates, df2.p_w, "--", color=action_colors["w"],    label="$p_w$")
        ax2r.plot(df2.dates, df2.p_s, "--", color=action_colors["s"],  label="$p_s$")
        ax2r.plot(df2.dates, df2.p_l, "--", color=action_colors["l"],   label="$p_l$")
        ax2r.set_ylim(-0.05, 1.05)

    # ------- single top legend ------------------------------------------
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.935),
                   ncol=6, frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])   # leave space for legend/suptitle
    if save_dir:
        plt.savefig(os.path.join(
            save_dir,
            f"compare_{measure1}+{measure2}_trajectories.jpg"
        ), dpi=dpi)
    else:
        plt.show()


def compare_measures_multiple_trajectories_by_pids(
        measure1: str,
        measure2: str,
        model_paths_dir: str,
        budget: int,
        pid1_list: list[int],
        pid2_list: list[int],
        save_dir: str | None = None,
):
    """
    Compare epidemic trajectories for *measure1* vs *measure2* for explicit
    lists of policy IDs. One row per (pid1, pid2) pair.

    A policy ID of –1 means “most recent” for that measure/budget combo.

    The legend appears **once** at the top of the figure.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    set_params_policies()
    if len(pid1_list) != len(pid2_list):
        raise ValueError("pid1_list and pid2_list must have the same length")

    # Helper to resolve -1 → latest numeric sub‑folder
    def _resolve_pid(trans_dir: str, pid_hint: int) -> int:
        if pid_hint != -1:
            return pid_hint
        nums = [int(d) for d in os.listdir(trans_dir) if d.isdigit()]
        if not nums:
            raise FileNotFoundError(f"No numeric subdirs in {trans_dir}")
        return max(nums)

    tdir1 = os.path.join(model_paths_dir,
                         f"{measure1}_results", f"budget_{budget}", "policies-transitions")
    tdir2 = os.path.join(model_paths_dir,
                         f"{measure2}_results", f"budget_{budget}", "policies-transitions")

    n_rows = len(pid1_list)
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(28, 5 * n_rows),
                             sharey=True, squeeze=False)
    date_fmt = DateFormatter("%d/%m")

    fig.suptitle(f"{measure_mapping[measure1]} vs {measure_mapping[measure2]}",
                 fontsize=28, y=0.97)

    # Storage for top legend
    legend_handles: list = []
    legend_labels: list = []

    for r, (pid1_hint, pid2_hint) in enumerate(zip(pid1_list, pid2_list)):
        pid1 = _resolve_pid(tdir1, pid1_hint)
        pid2 = _resolve_pid(tdir2, pid2_hint)

        def _load_csv(tdir: str, pid: int) -> pd.DataFrame:
            return pd.read_csv(os.path.join(tdir, str(pid), "run_0.csv"),
                               parse_dates=["dates"])

        df1, df2 = _load_csv(tdir1, pid1), _load_csv(tdir2, pid2)
        ax_l, ax_r = axes[r]

        # ---- left -------------------------------------------------------
        ax_l.plot(df1.dates, df1.i_hosp_new, color="deepskyblue", label="daily new hosp")
        ax_l.plot(df1.dates, df1.i_icu_new, color="orange",    label="daily new ICU")
        ax_l.plot(df1.dates, df1.d_new,       color="red",      label="daily deaths")
        ax_l.set_title(f"Policy {pid1} for {measure_mapping[measure1]}",
                       fontweight="normal")
        if r == n_rows - 1:
            ax_l.set_xlabel("Date")
        ax_l.set_ylabel("Individuals")
        ax_l.set_ylim(0, 4500)
        ax_l.xaxis.set_major_formatter(date_fmt)
        ax_l.tick_params(axis="x", rotation=45)

        ax2l = ax_l.twinx()
        ax2l.plot(df1.dates, df1.p_w, "--", color=action_colors["w"],    label="$p_w$")
        ax2l.plot(df1.dates, df1.p_s, "--", color=action_colors["s"],  label="$p_s$")
        ax2l.plot(df1.dates, df1.p_l, "--", color=action_colors["l"],   label="$p_l$")
        ax2l.set_ylim(-0.05, 1.05)

        # Capture legend items once
        if r == 0:
            h1, l1 = ax_l.get_legend_handles_labels()
            h2, l2 = ax2l.get_legend_handles_labels()
            legend_handles = h1 + h2
            legend_labels = l1 + l2

        # ---- right ------------------------------------------------------
        ax_r.plot(df2.dates, df2.i_hosp_new, color="deepskyblue", label="daily new hosp")
        ax_r.plot(df2.dates, df2.i_icu_new, color="orange",    label="daily new ICU")
        ax_r.plot(df2.dates, df2.d_new,       color="red",      label="daily deaths")
        ax_r.set_title(f"Policy {pid2} for {measure_mapping[measure2]}",
                       fontweight="normal")
        if r == n_rows - 1:
            ax_r.set_xlabel("Date")
        ax_r.xaxis.set_major_formatter(date_fmt)
        ax_r.tick_params(axis="x", rotation=45)

        ax2r = ax_r.twinx()
        ax2r.plot(df2.dates, df2.p_w, "--", color=action_colors["w"],    label="$p_w$")
        ax2r.plot(df2.dates, df2.p_s, "--", color=action_colors["s"],  label="$p_s$")
        ax2r.plot(df2.dates, df2.p_l, "--", color=action_colors["l"],   label="$p_l$")
        ax2r.set_ylim(-0.05, 1.05)

    # -------- top legend ------------------------------------------------
    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc="upper center", bbox_to_anchor=(0.5, 0.928),
                   ncol=6, frameon=False, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_dir:
        fname = f"compare_{measure1}+{measure2}_trajectories_{pid1_list}+{pid2_list}.jpg"
        plt.savefig(os.path.join(save_dir, fname.replace(' ', '')), dpi=dpi)
    else:
        plt.show()


if __name__ == "__main__":
    budget = 5

    all_measures = {
        # "sb": 9,
        # "sb_sbs": 2,
        # "sbs": 5,
        "abfta": 8,
        "sbs_abfta": 0
    }
    save_dir = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/"
    #
    plot_hospitalization_risk_bar(age_groups, hospitalization_risks, save_dir=save_dir)

    for m, s in all_measures.items():
        repr_policies = radar_plots(m, test=True)
        print(repr_policies)

        measure_pcn = m
        model_paths_dir = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/{measure_pcn}_results"
        scales = [800000, 11000, 50., 20, 50, 120, 24e4, 0.08]
        #scales = [800000, 11000, 50., 20, 50, 120, 24e4, 1]
        scale = np.array(scales)
        make_budget_plots(measure_pcn, scale, model_paths_dir, repr_policies, save_dir=save_dir, interactive=False)

    # run_episodes = False
    # pids = []
    # goal_hospitalizations = 5000
    # for measure_pcn, seed in all_measures.items():
    #     print(f"Finding policy for {measure_pcn}")
    #     save_dir = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/"
    #     policy_id = main(measure_pcn, goal_hospitalizations, seed, save_dir, run_episodes)
    #     pids.append(policy_id)
    #
    results_path = f"/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results/"

    # for i in range(len(age_groups)):
    #     plot_age_group_comparison(pids, i, all_measures, results_path, budget)

    # plot_age_group_grid_comparison([2000, 5000, 8000], all_measures, results_path, budget, save_dir=save_dir)

    #plot_two_age_groups_comparison([2000, 5000, 8000], list(all_measures.keys()), results_path, budget, (2, 7), "lower right", save_dir=save_dir)
    #plot_two_age_groups_comparison([2000, 5000, 8000], list(all_measures.keys()), results_path, budget, (1, 3), "lower right", save_dir=save_dir)



    # compare_two_measures_trajectories(
    #     "sb", "sb_sbs",
    #     results_path,
    #     pid1=None,
    #     pid2=None,
    #     budget=5,
    #     goal_hospitalizations=goal_hospitalizations
    # )


    # combinations = [
    #     ("sb", "sb_sbs"),
    #     ("sb_sbs", "sbs"),
    #     ("sbs", "abfta"),
    #     ("sbs", "sbs_abfta")
    # ]
    #
    # for measure1, measure2 in combinations:
    #     compare_two_measures_multiple_trajectories(
    #         measure1=measure1, measure2=measure2,
    #         model_paths_dir=results_path,
    #         budget=5, save_dir=results_path,
    #         goal_hospitalizations_list=[2000, 5000, 8000]
    #     )
    #
    #     compare_measures_multiple_trajectories_by_pids(
    #         measure1=measure1, measure2=measure2,
    #         model_paths_dir=results_path,
    #         budget=5, pid1_list=[0, -1], pid2_list=[0, -1],
    #         save_dir=results_path
    #     )
    #
    #     # --- NEW: heat‑map only comparisons (fairness objectives) -----------
    #     if ("abfta" in (measure1, measure2)) or ("sbs_abfta" in (measure1, measure2)):
    #         compare_two_measures_heatmaps(
    #             measure1=measure1, measure2=measure2,
    #             model_paths_dir=results_path,
    #             budget=5, save_dir=results_path,
    #             goal_hospitalizations_list=[2000, 5000, 8000]
    #         )
    #
    #         compare_measures_heatmaps_by_pids(
    #             measure1=measure1, measure2=measure2,
    #             model_paths_dir=results_path,
    #             budget=5, pid1_list=[0, -1], pid2_list=[0, -1],
    #             save_dir=results_path
    #         )
