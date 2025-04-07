import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from agent.pcn.pcn import choose_action
from scenario.create_fair_env import *
from scenario.pcn_model import *

# -------------------------------
# Keep your scaling parameters:
scale = np.array([800000, 10000, 50., 20, 50, 100])
ref_point = np.array([-15000000, -200000, -1000., -1000., -1000., -1000.]) / scale
max_return = np.array([0, 0, 0, 0, 0, 0]) / scale

MODEL_PATH = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/fair_covid_2/experiments/results/cluster/steps_300000/objectives_R_ARH:R_SB_W:R_SB_S:R_SB_L_SBS:ABFTA/distance_metric_none/seed_0/6obj_3days_crashed/model_9.pt"  # your single best model
N_RUNS = 100
OUTPUT_CSV = "pareto_points_3.csv"

import argparse

def get_default_args():
    parser = argparse.ArgumentParser()
    from scenario.create_fair_env import fMDP_parser
    return fMDP_parser.parse_args([])  # ← this gives you the default args

args = get_default_args()


def run_policy_with_params(model, desired_return, desired_horizon):
    env, ref_point, scaling_factor, max_return, ss, se, sa, nA, with_budget = create_fair_covid_env(args, [0, 1, 2, 3, 4, 5])
    env.reset()

    done = False
    totalHosp = 0.0
    totalLostContacts = 0.0

    while not done:
        state_df, C_diff_6x10x10 = env.state_df()

        # sum daily new hospital admissions
        daily_hosp = state_df["I_hosp_new"].sum() if "I_hosp_new" in state_df else 0.0
        totalHosp += daily_hosp

        # sum lost contacts from the 6x10x10 matrix
        if C_diff_6x10x10 is not None:
            totalLostContacts += env.lost_contacts.sum()

        budget = np.ones(3) * 4
        action_so_far = env.current_action
        events = env.current_events_n
        state = env.current_state_n


        action = choose_action(
            model,
            (budget, state, events, action_so_far),
            desired_return,
            desired_horizon,
            eval=True
        )

        _, _, done, info = env.step(action)

    return totalHosp, totalLostContacts


def main():
    # load your single best model
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

    rows = []
    returns = pd.read_csv("wandb_export_2025-03-24T13_35_19.075+01_00.csv")

    for i in range(min(N_RUNS, len(returns))):
        row = returns.iloc[i].values.astype(np.float32)
        desired_return = row  # shape: (num_objectives,) #np.array(returns[i]).astype(np.float32)
        desired_return = np.random.uniform(low=np.array([0, 0, 0, 0, 0, 0]), high=scale).astype(np.float32)
        # Also vary horizon if you want
        desired_horizon = 15 #np.random.randint(1, 12)  # e.g. random in [1..12]

        hosp, lostC = run_policy_with_params(model, desired_return, desired_horizon)
        rows.append({
            "desired_return": desired_return.tolist(),
            "desired_horizon": float(desired_horizon),
            "totalHosp": hosp,
            "totalLostContacts": lostC,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} points to {OUTPUT_CSV}")

    xvals = -df["totalHosp"] / 1e5
    yvals = df["totalLostContacts"]  # might be negative, so it goes downward
    xvals, yvals = np.array(xvals, yvals) * np.array([[10000], [100]])
    plt.scatter(xvals, yvals, marker="o", color="blue")

    plt.xlabel("Cumulative number of daily new hospitalizations ×10^5 (negated)")
    plt.ylabel("Cumulative lost contacts")
    plt.title("Approx. Pareto front from model_10 with random desired_returns")
    plt.savefig("pareto_front.png")

def plot_pareto_points():
    df = pd.read_csv(OUTPUT_CSV)
    xvals = df["totalHosp"] / 1e5
    yvals = -df["totalLostContacts"]  # might be negative, so it goes downward
    #xvals, yvals = np.array(xvals, yvals) * np.array([[10000], [100]])
    plt.scatter(xvals, yvals, marker="o", color="blue")

    plt.xlabel("Cumulative number of daily new hospitalizations ×10^5 (negated)")
    plt.ylabel("Cumulative lost contacts")
    plt.title("Approx. Pareto front from model_10 with random desired_returns")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_pareto_fronts():
    csv_files = ["sbs.csv", "sbs1.csv", "sbs2.csv", "sbs3.csv"]

    plt.figure(figsize=(10, 7))

    # Plot the fixed data
    df_fixed = pd.read_csv("fixed.csv")
    x_fixed = df_fixed["o_0"]
    y_fixed = df_fixed["o_1"]
    plt.scatter(x_fixed, y_fixed, s=5, alpha=0.7, label="fixed", marker='o')

    # Plot each CSV in a loop
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        x = df["o_0"]
        y = df["o_1"]
        print(df["o_2"].min())
        plt.scatter(x, y, s=5, alpha=0.7, label=csv_file, marker='o')

    plt.ylim(-2000, 0)  # Adjust as needed
    plt.xlabel("X-axis values")
    plt.ylabel("Y-axis values")
    plt.title("Scatter Plot with Small Markers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pareto_fronts.png")

if __name__ == "__main__":
    #plot_pareto_points()
    plot_pareto_fronts()

    #main()