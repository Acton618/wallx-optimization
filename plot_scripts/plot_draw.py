"""
plot_flow.py
============
Visualize Flow Matching denoising trajectories for robotic actions.

Inputs
------
noisy_action_trajectory.txt
    comma-separated, 7 columns (no header):
    experiment,step,time,batch,action_timestep,dim,value

Outputs
-------
final_action_sim{sim_id}.png
    Final denoised action as a mechanical arm control curve.

denoising_sim{sim_id}.png
    10 subplots (action horizon t+0 ... t+9), each showing
    how the 7 real action dims evolve during denoising.

final_action_all_sims.png
    One consolidated robot control curve combining final action chunks
    from sim_0 ... sim_6 (global action timeline).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config (must match your dof_config / action_horizon)
# ----------------------------------------------------------------------
ACTION_HORIZON = 10
ACTION_DIM = 26

REAL_DIMS = {
    "Δpos_x": 0,
    "Δpos_y": 1,
    "Δpos_z": 2,
    "Δrot_z": 3,
    "Δrot_y": 4,
    "Δrot_x": 5,
    "gripper": 6,
}

COLUMNS = [
    "experiment",
    "step",
    "time",
    "batch",
    "action_timestep",
    "dim",
    "value",
]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(
        path,
        sep=",",
        header=None,
        names=COLUMNS,
    )
    df["step"] = df["step"].astype(int)
    df["action_timestep"] = df["action_timestep"].astype(int)
    df["dim"] = df["dim"].astype(int)
    df["value"] = df["value"].astype(float)
    df["time"] = df["time"].astype(float)
    return df


def get_sim_ids(df):
    """Return sorted list of sim ids, e.g. [0,1,2,3,4,5,6]"""
    sims = df["experiment"].unique()
    ids = []
    for s in sims:
        if s.startswith("sim_"):
            try:
                ids.append(int(s.split("_")[1]))
            except ValueError:
                pass
    return sorted(ids)


# ----------------------------------------------------------------------
# Tensor reconstruction
# ----------------------------------------------------------------------
def get_noisy_tensor(df, sim, step):
    """
    Return shape [ACTION_HORIZON, ACTION_DIM]
    step = ODE step (final action uses step=10)
    """
    sub = df[(df["experiment"] == f"sim_{sim}") & (df["step"] == step)]
    out = np.zeros((ACTION_HORIZON, ACTION_DIM))
    for _, row in sub.iterrows():
        tt = int(row["action_timestep"])
        d = int(row["dim"])
        out[tt, d] = row["value"]
    return out


# ----------------------------------------------------------------------
# Figure 1: final denoised action -> mechanical arm control curve
# ----------------------------------------------------------------------
def plot_final_action(df, sim_id, save=True):
    final_action = get_noisy_tensor(df, sim=sim_id, step=10)  # [10, 26]

    plt.figure(figsize=(12, 6))
    for name, d in REAL_DIMS.items():
        plt.plot(range(ACTION_HORIZON), final_action[:, d], marker="o", label=name)

    plt.title(f"Final Denoised Action (sim_{sim_id})\nMechanical Arm Control Curve")
    plt.xlabel("Action Horizon Step (t+0 ... t+9)")
    plt.ylabel("Action Value (normalized)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(f"final_action_sim{sim_id}.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Figure 2: denoising process inside one sim (10 parallel subplots)
# ----------------------------------------------------------------------
def plot_denoising_per_action_step(df, sim_id, save=True):
    fig, axes = plt.subplots(2, 5, figsize=(25, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for tt in range(ACTION_HORIZON):
        ax = axes[tt]
        for name, d in REAL_DIMS.items():
            ys = []
            steps = list(range(11))  # step 0..10
            for s in steps:
                sub = df[
                    (df["experiment"] == f"sim_{sim_id}")
                    & (df["step"] == s)
                    & (df["action_timestep"] == tt)
                    & (df["dim"] == d)
                ]
                if len(sub) > 0:
                    ys.append(sub["value"].values[0])
                else:
                    ys.append(np.nan)
            ax.plot(steps, ys, marker="o", label=name)

        ax.set_title(f"action step t+{tt}")
        ax.grid(True)
        if tt >= 5:
            ax.set_xlabel("Flow step (0 = noise, 10 = final)")
        if tt % 5 == 0:
            ax.set_ylabel("Action value")

    axes[9].legend(
        REAL_DIMS.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
    )
    fig.suptitle(
        f"Denoising Process inside sim_{sim_id}\n(one curve per action dim)",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    if save:
        fig.savefig(f"denoising_sim{sim_id}.png", dpi=120, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------
# Figure 3: all sims combined -> one consolidated robot control curve
# ----------------------------------------------------------------------
def plot_all_sims_combined(df, sim_ids, save=True):
    """
    Concatenate final action chunks from sim_0 ... sim_N along action horizon.
    Shape: [len(sim_ids) * ACTION_HORIZON, ACTION_DIM]
    """
    chunks = []
    for sim_id in sim_ids:
        chunks.append(get_noisy_tensor(df, sim=sim_id, step=10))
    combined = np.concatenate(chunks, axis=0)  # [N_sim*10, 26]

    x = np.arange(len(sim_ids) * ACTION_HORIZON)

    plt.figure(figsize=(16, 8))
    for name, d in REAL_DIMS.items():
        plt.plot(x, combined[:, d], marker="o", markersize=3, label=name)

    # 在每个 sim 边界画分隔线
    for i in range(1, len(sim_ids)):
        plt.axvline(i * ACTION_HORIZON - 0.5, color="gray", linestyle="--", alpha=0.5)

    plt.title("Consolidated Robot Control Curve (sim_0 ... sim_N)\nFinal Denoised Action Chunks")
    plt.xlabel("Global Action Step (sim boundaries shown)")
    plt.ylabel("Action Value (normalized)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig("final_action_all_sims.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Plot Flow Matching denoising trajectories.")
    parser.add_argument("--file", type=str, default="/opt/data/zhuoziying/wall-x/debug_output/noisy_action_trajectory.txt",
                        help="Path to noisy_action_trajectory.txt")
    parser.add_argument("--sim", type=int, default=0,
                        help="Which sim to visualize (e.g. 0 for sim_0)")
    parser.add_argument("--all_sims", action="store_true",
                        help="Auto-process sim_0 ... sim_N")
    args = parser.parse_args()

    df = load_data(args.file)
    sim_ids = get_sim_ids(df)

    if args.all_sims:
        for sim_id in sim_ids:
            print(f"[INFO] Processing sim_{sim_id} ...")
            plot_final_action(df, sim_id=sim_id, save=True)
            plot_denoising_per_action_step(df, sim_id=sim_id, save=True)

        # 额外画一张“所有 sim 最终 action chunk 合并”的图
        plot_all_sims_combined(df, sim_ids=sim_ids, save=True)
    else:
        plot_final_action(df, sim_id=args.sim, save=True)
        plot_denoising_per_action_step(df, sim_id=args.sim, save=True)


if __name__ == "__main__":
    main()