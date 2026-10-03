"""
plot_flow.py
============
Visualize Flow Matching denoising trajectories for robotic actions.

State layout after reconstruction:
    all_states[chunk_id, state_step, action_id, dim_id]

where:
    state_step = 0  -> CSV step=0  -> x_t0
    state_step = 1  -> CSV step=1  -> x_t1
    ...
    state_step = 9  -> CSV step=9  -> x_t9
    state_step = 10 -> CSV step=10 -> x_t10, final output

Therefore:
    final denoising input  = all_states[:, 9]
    final denoising output = all_states[:, 10]
    final denoising delta  = all_states[:, 10] - all_states[:, 9]

Outputs:
    fig1_overall_sequence.png
    fig2_chunk_action_heatmap.png
    fig3_single_chunk_denoising_lines_chunkX.png
    fig4_single_chunk_denoising_heatmap_chunkX.png
    fig5_mean_abs_change.png
    fig6_change_heatmap.png
    fig7_chunk_boundary_jump.png
    fig8_padding_check.png
    fig9_final_denoising_input_output_chunkX.png
    fig9b_final_denoising_change_heatmap.png
    fig10_final_denoising_delta_chunkX.png
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
NUM_CHUNKS = 7
NUM_DENOISE_STEPS = 10       # CSV step=0..9
FINAL_STEP = NUM_DENOISE_STEPS  # CSV step=10

ACTIONS_PER_CHUNK = 10
ACTION_DIM = 26

VALID_DIM = 7                # pos3 + rot3 + gripper1
PADDING_DIM = ACTION_DIM - VALID_DIM

REAL_DIMS = {
    "dpos_x": 0,
    "dpos_y": 1,
    "dpos_z": 2,
    "drot_z": 3,
    "drot_y": 4,
    "drot_x": 5,
    "gripper": 6,
}

DIM_NAMES = [
    "dpos_x",
    "dpos_y",
    "dpos_z",
    "drot_z",
    "drot_y",
    "drot_x",
    "gripper",
]

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
    sims = df["experiment"].unique()

    ids = []
    for sim_name in sims:
        if not isinstance(sim_name, str):
            continue

        if sim_name.startswith("sim_"):
            try:
                ids.append(int(sim_name.split("_")[1]))
            except ValueError:
                pass

    return sorted(ids)


# ----------------------------------------------------------------------
# Data validation
# ----------------------------------------------------------------------
def validate_data(df, sim_ids):
    """
    Check whether each sim and each step contains the expected rows.

    Expected rows per step:
        ACTIONS_PER_CHUNK * ACTION_DIM
    """

    expected_rows = ACTIONS_PER_CHUNK * ACTION_DIM
    all_steps = list(range(NUM_DENOISE_STEPS + 1))

    print("\n[INFO] Validating input data...")
    print(f"[INFO] Expected rows per sim/step: {expected_rows}")

    for sim_id in sim_ids:
        sim_name = f"sim_{sim_id}"
        sim_df = df[df["experiment"] == sim_name]

        for step in all_steps:
            step_df = sim_df[sim_df["step"] == step]
            row_count = len(step_df)

            if row_count != expected_rows:
                print(
                    f"[WARNING] {sim_name}, step={step}: "
                    f"expected {expected_rows} rows, "
                    f"got {row_count}"
                )

            invalid_action_timestep = step_df[
                (step_df["action_timestep"] < 0)
                | (step_df["action_timestep"] >= ACTIONS_PER_CHUNK)
            ]

            if len(invalid_action_timestep) > 0:
                print(
                    f"[WARNING] {sim_name}, step={step}: "
                    f"found invalid action_timestep values"
                )

            invalid_dim = step_df[
                (step_df["dim"] < 0)
                | (step_df["dim"] >= ACTION_DIM)
            ]

            if len(invalid_dim) > 0:
                print(
                    f"[WARNING] {sim_name}, step={step}: "
                    f"found invalid dim values"
                )

            duplicated = step_df.duplicated(
                subset=["action_timestep", "dim"]
            ).sum()

            if duplicated > 0:
                print(
                    f"[WARNING] {sim_name}, step={step}: "
                    f"found {duplicated} duplicated "
                    f"(action_timestep, dim) rows"
                )

    print("[INFO] Data validation finished.\n")


# ----------------------------------------------------------------------
# Tensor reconstruction
# ----------------------------------------------------------------------
def build_denoising_tensor(df, sim_ids):
    """
    Build states for CSV step=0..9.

    Returns:
        action_denoising:
            [num_chunks, NUM_DENOISE_STEPS, ACTIONS_PER_CHUNK, ACTION_DIM]
    """

    num_chunks = len(sim_ids)

    out = np.zeros(
        (
            num_chunks,
            NUM_DENOISE_STEPS,
            ACTIONS_PER_CHUNK,
            ACTION_DIM,
        ),
        dtype=np.float64,
    )

    for chunk_id, sim_id in enumerate(sim_ids):
        sim_name = f"sim_{sim_id}"

        for step in range(NUM_DENOISE_STEPS):
            sub = df[
                (df["experiment"] == sim_name)
                & (df["step"] == step)
            ]

            for _, row in sub.iterrows():
                action_timestep = int(row["action_timestep"])
                dim = int(row["dim"])

                if not (
                    0 <= action_timestep < ACTIONS_PER_CHUNK
                ):
                    continue

                if not (0 <= dim < ACTION_DIM):
                    continue

                out[
                    chunk_id,
                    step,
                    action_timestep,
                    dim,
                ] = row["value"]

    return out


def get_final_action_chunks(df, sim_ids):
    """
    Build final output for CSV step=10.

    Returns:
        final_chunks:
            [num_chunks, ACTIONS_PER_CHUNK, ACTION_DIM]
    """

    chunks = []

    for sim_id in sim_ids:
        sim_name = f"sim_{sim_id}"

        sub = df[
            (df["experiment"] == sim_name)
            & (df["step"] == FINAL_STEP)
        ]

        out = np.zeros(
            (
                ACTIONS_PER_CHUNK,
                ACTION_DIM,
            ),
            dtype=np.float64,
        )

        for _, row in sub.iterrows():
            action_timestep = int(row["action_timestep"])
            dim = int(row["dim"])

            if not (
                0 <= action_timestep < ACTIONS_PER_CHUNK
            ):
                continue

            if not (0 <= dim < ACTION_DIM):
                continue

            out[action_timestep, dim] = row["value"]

        chunks.append(out)

    if len(chunks) == 0:
        raise ValueError(
            f"No final action data found for step={FINAL_STEP}"
        )

    return np.stack(chunks, axis=0)


def build_full_state_tensor(action_denoising, final_chunks):
    """
    Concatenate step=0..9 and step=10.

    Input:
        action_denoising:
            [C, 10, A, D]

        final_chunks:
            [C, A, D]

    Returns:
        all_states:
            [C, 11, A, D]

    Meaning:
        all_states[:, 0]  -> step=0  -> x_t0
        all_states[:, 1]  -> step=1  -> x_t1
        ...
        all_states[:, 9]  -> step=9  -> x_t9
        all_states[:, 10] -> step=10 -> x_t10
    """

    if action_denoising.shape[0] != final_chunks.shape[0]:
        raise ValueError(
            "Chunk count mismatch between action_denoising "
            "and final_chunks"
        )

    final_state = final_chunks[:, None, :, :]

    all_states = np.concatenate(
        [
            action_denoising,
            final_state,
        ],
        axis=1,
    )

    return all_states


# ----------------------------------------------------------------------
# Figure 1: overall final action sequence
# ----------------------------------------------------------------------
def fig1_overall_sequence(final_chunks):
    """
    final_chunks:
        [C, A, D]
    """

    full = final_chunks[:, :, :VALID_DIM].reshape(
        -1,
        VALID_DIM,
    )

    n_steps = full.shape[0]

    fig, axes = plt.subplots(
        VALID_DIM,
        1,
        figsize=(14, 2.2 * VALID_DIM),
        sharex=True,
    )

    if VALID_DIM == 1:
        axes = [axes]

    for dim in range(VALID_DIM):
        axes[dim].plot(
            np.arange(n_steps),
            full[:, dim],
            marker=".",
            linewidth=1.5,
        )

        for x in range(
            ACTIONS_PER_CHUNK,
            n_steps,
            ACTIONS_PER_CHUNK,
        ):
            axes[dim].axvline(
                x,
                color="gray",
                linestyle="--",
                alpha=0.5,
            )

        axes[dim].set_ylabel(DIM_NAMES[dim])
        axes[dim].grid(alpha=0.3)

    axes[-1].set_xlabel("Global action step")

    fig.suptitle(
        "Figure 1: Overall Final Action Sequence",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        "fig1_overall_sequence.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 2: final chunk-action heatmap
# ----------------------------------------------------------------------
def fig2_chunk_action_heatmap(final_chunks):
    """
    final_chunks:
        [C, A, D]
    """

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8),
    )

    axes = axes.flatten()

    for dim in range(VALID_DIM):
        data = final_chunks[:, :, dim]

        im = axes[dim].imshow(
            data,
            aspect="auto",
            cmap="coolwarm",
        )

        axes[dim].set_title(DIM_NAMES[dim])
        axes[dim].set_xlabel("Action index in chunk")
        axes[dim].set_ylabel("Chunk index")

        fig.colorbar(
            im,
            ax=axes[dim],
            fraction=0.046,
            pad=0.04,
        )

    for dim in range(VALID_DIM, len(axes)):
        axes[dim].axis("off")

    fig.suptitle(
        "Figure 2: Final Chunk-Action Heatmap",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        "fig2_chunk_action_heatmap.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 3: single chunk state trajectories
# ----------------------------------------------------------------------
def fig3_single_chunk_denoising_lines(
    action_denoising,
    chunk_id=0,
):
    """
    Plot saved states step=0..9.

    action_denoising:
        [C, 10, A, D]
    """

    num_chunks = action_denoising.shape[0]

    if not (0 <= chunk_id < num_chunks):
        raise ValueError(
            f"Invalid chunk_id={chunk_id}. "
            f"Available range: 0..{num_chunks - 1}"
        )

    data = action_denoising[
        chunk_id,
        :,
        :,
        :VALID_DIM,
    ]

    fig, axes = plt.subplots(
        VALID_DIM,
        1,
        figsize=(12, 2.6 * VALID_DIM),
        sharex=True,
    )

    if VALID_DIM == 1:
        axes = [axes]

    state_ids = np.arange(NUM_DENOISE_STEPS)

    for dim in range(VALID_DIM):
        for action_id in range(ACTIONS_PER_CHUNK):
            axes[dim].plot(
                state_ids,
                data[:, action_id, dim],
                marker="o",
                alpha=0.8,
                label=f"act {action_id}",
            )

        axes[dim].set_ylabel(DIM_NAMES[dim])
        axes[dim].grid(alpha=0.3)

    axes[0].legend(
        ncol=5,
        fontsize=8,
    )

    axes[-1].set_xlabel(
        "Saved state step (0 ... 9)"
    )

    fig.suptitle(
        f"Figure 3: Chunk {chunk_id} "
        "Denoising State Trajectories",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        f"fig3_single_chunk_denoising_lines_chunk{chunk_id}.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 4: single chunk state heatmap
# ----------------------------------------------------------------------
def fig4_single_chunk_denoising_heatmap(
    action_denoising,
    chunk_id=0,
):
    """
    Plot saved states step=0..9.

    data[:, :, dim].T:
        [action_id, saved_state_step]
    """

    num_chunks = action_denoising.shape[0]

    if not (0 <= chunk_id < num_chunks):
        raise ValueError(
            f"Invalid chunk_id={chunk_id}. "
            f"Available range: 0..{num_chunks - 1}"
        )

    data = action_denoising[
        chunk_id,
        :,
        :,
        :VALID_DIM,
    ]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(18, 8),
    )

    axes = axes.flatten()

    for dim in range(VALID_DIM):
        im = axes[dim].imshow(
            data[:, :, dim].T,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
        )

        axes[dim].set_title(DIM_NAMES[dim])
        axes[dim].set_xlabel("Saved state step")
        axes[dim].set_ylabel("Action index in chunk")

        axes[dim].set_xticks(
            np.arange(NUM_DENOISE_STEPS)
        )

        fig.colorbar(
            im,
            ax=axes[dim],
            fraction=0.046,
            pad=0.04,
        )

    for dim in range(VALID_DIM, len(axes)):
        axes[dim].axis("off")

    fig.suptitle(
        f"Figure 4: Chunk {chunk_id} "
        "Denoising State Heatmap",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        f"fig4_single_chunk_denoising_heatmap_chunk{chunk_id}.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 5: mean absolute change including final transition
# ----------------------------------------------------------------------
def fig5_mean_abs_change(all_states):
    """
    all_states:
        [C, 11, A, D]

    Transitions:
        step=0 -> step=1
        ...
        step=8 -> step=9
        step=9 -> step=10
    """

    valid = all_states[:, :, :, :VALID_DIM]

    delta = np.diff(
        valid,
        axis=1,
    )  # [C, 10, A, VALID_DIM]

    mean_abs_delta = np.mean(
        np.abs(delta),
        axis=(2, 3),
    )  # [C, 10]

    transition_labels = [
        "0→1",
        "1→2",
        "2→3",
        "3→4",
        "4→5",
        "5→6",
        "6→7",
        "7→8",
        "8→9",
        "9→10",
    ]

    plt.figure(figsize=(11, 5))

    for chunk_id in range(mean_abs_delta.shape[0]):
        plt.plot(
            np.arange(1, 11),
            mean_abs_delta[chunk_id],
            marker="o",
            label=f"Chunk {chunk_id}",
        )

    plt.xlabel("Denoising transition")
    plt.ylabel("Mean absolute change")
    plt.xticks(
        np.arange(1, 11),
        transition_labels,
        rotation=45,
    )
    plt.legend()
    plt.grid(alpha=0.3)

    plt.title(
        "Figure 5: Mean Absolute Change "
        "Including Final Transition",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        "fig5_mean_abs_change.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 6: change heatmap including final transition
# ----------------------------------------------------------------------
def fig6_change_heatmap(all_states):
    """
    Heatmap over chunk and denoising transition.
    """

    valid = all_states[:, :, :, :VALID_DIM]

    delta = np.diff(
        valid,
        axis=1,
    )

    mean_abs_delta = np.mean(
        np.abs(delta),
        axis=(2, 3),
    )  # [C, 10]

    transition_labels = [
        "0→1",
        "1→2",
        "2→3",
        "3→4",
        "4→5",
        "5→6",
        "6→7",
        "7→8",
        "8→9",
        "9→10",
    ]

    plt.figure(figsize=(11, 5))

    im = plt.imshow(
        mean_abs_delta,
        aspect="auto",
        cmap="viridis",
    )

    plt.xlabel("Denoising transition")
    plt.ylabel("Chunk index")

    plt.xticks(
        np.arange(10),
        transition_labels,
        rotation=45,
    )

    plt.colorbar(
        im,
        label="Mean absolute change",
    )

    plt.title(
        "Figure 6: Denoising Change Heatmap "
        "Including Final Transition",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        "fig6_change_heatmap.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 7: chunk boundary jump
# ----------------------------------------------------------------------
def fig7_chunk_boundary_jump(final_chunks):
    """
    Boundary jump between:
        chunk c-1, last action
        chunk c, first action
    """

    valid = final_chunks[:, :, :VALID_DIM]

    num_chunks = valid.shape[0]
    jumps = []

    for chunk_id in range(1, num_chunks):
        previous_last = valid[
            chunk_id - 1,
            -1,
            :,
        ]

        current_first = valid[
            chunk_id,
            0,
            :,
        ]

        jumps.append(
            np.linalg.norm(
                current_first - previous_last
            )
        )

    jumps = np.asarray(jumps)

    plt.figure(figsize=(8, 5))

    plt.plot(
        np.arange(1, num_chunks),
        jumps,
        marker="o",
    )

    plt.xlabel(
        "Chunk boundary "
        "(between chunk c-1 and chunk c)"
    )
    plt.ylabel(
        "Boundary jump "
        "(L2, valid dims)"
    )

    plt.grid(alpha=0.3)

    plt.title(
        "Figure 7: Chunk Boundary Continuity",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        "fig7_chunk_boundary_jump.png",
        dpi=150,
    )
    plt.show()


# ----------------------------------------------------------------------
# Figure 8: padding dimension check
# ----------------------------------------------------------------------
def fig8_padding_check(
    action_denoising,
    final_chunks,
):
    """
    Check padding dimensions.

    Denoising states:
        CSV step=0..9

    Final output:
        CSV step=10
    """

    pad_denoise = action_denoising[
        :, :, :, VALID_DIM:
    ]

    max_pad_denoise = np.max(
        np.abs(pad_denoise),
        axis=(2, 3),
    )  # [C, 10]

    pad_final = final_chunks[
        :, :, VALID_DIM:
    ]

    max_pad_final = np.max(
        np.abs(pad_final),
        axis=(1, 2),
    )  # [C]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    # 左图：每个 chunk 的 denoising padding
    for chunk_id in range(max_pad_denoise.shape[0]):
        axes[0].plot(
            np.arange(NUM_DENOISE_STEPS),
            max_pad_denoise[chunk_id],
            marker="o",
            label=f"Chunk {chunk_id}",
        )

    axes[0].set_title(
        "Padding during denoising states"
    )
    axes[0].set_xlabel(
        "Saved state step (0 ... 9)"
    )
    axes[0].set_ylabel(
        "Max |padding value|"
    )
    axes[0].set_xticks(
        np.arange(NUM_DENOISE_STEPS)
    )
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # 右图：最终输出 padding
    axes[1].plot(
        np.arange(max_pad_final.shape[0]),
        max_pad_final,
        marker="s",
        linewidth=2,
        color="red",
    )

    axes[1].set_title(
        "Padding in final output"
    )
    axes[1].set_xlabel("Chunk index")
    axes[1].set_ylabel(
        "Max |padding value|"
    )
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Figure 8: Padding Dimension Check",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        "fig8_padding_check.png",
        dpi=150,
    )
    plt.show()

# ----------------------------------------------------------------------
# Figure 9: final denoising input vs output
# ----------------------------------------------------------------------
def fig9_final_denoising_input_vs_output(
    all_states,
    chunk_id=0,
):
    """
    Compare the input and output of the final denoising transition.

    all_states:
        [num_chunks, 11, actions_per_chunk, action_dim]

    State definition:
        all_states[:, 0]  -> step=0  -> x_t0
        all_states[:, 1]  -> step=1  -> x_t1
        ...
        all_states[:, 9]  -> step=9  -> x_t9
        all_states[:, 10] -> step=10 -> x_t10

    Final denoising transition:
        step=9 -> step=10
    """

    if all_states.ndim != 4:
        raise ValueError(
            "all_states must have shape "
            "[num_chunks, num_states, actions_per_chunk, action_dim], "
            f"but got {all_states.shape}"
        )

    num_chunks = all_states.shape[0]
    num_states = all_states.shape[1]
    num_actions = all_states.shape[2]
    action_dim = all_states.shape[3]

    if num_states < 11:
        raise ValueError(
            f"all_states must contain at least 11 states "
            f"(step=0..10), but got {num_states}"
        )

    if num_actions != ACTIONS_PER_CHUNK:
        print(
            f"[WARNING] Expected {ACTIONS_PER_CHUNK} actions per chunk, "
            f"but got {num_actions}"
        )

    if action_dim < VALID_DIM:
        raise ValueError(
            f"action_dim={action_dim} is smaller than "
            f"VALID_DIM={VALID_DIM}"
        )

    if not (0 <= chunk_id < num_chunks):
        raise ValueError(
            f"Invalid chunk_id={chunk_id}. "
            f"Available range: 0..{num_chunks - 1}"
        )

    # --------------------------------------------------------------
    # 最后一步：step=9 -> step=10
    # --------------------------------------------------------------
    last_input = all_states[:, 9, :, :VALID_DIM]
    last_output = all_states[:, 10, :, :VALID_DIM]
    last_delta = last_output - last_input

    # 当前 chunk 的数据
    chunk_input = last_input[chunk_id]      # [A, VALID_DIM]
    chunk_output = last_output[chunk_id]    # [A, VALID_DIM]
    chunk_delta = last_delta[chunk_id]      # [A, VALID_DIM]

    # --------------------------------------------------------------
    # 统计量
    # --------------------------------------------------------------

    # 当前 chunk 的整体平均绝对变化：scalar
    mean_abs_delta_chunk = np.mean(
        np.abs(chunk_delta)
    )

    # 当前 chunk 每个维度的平均绝对变化：[VALID_DIM]
    mean_abs_delta_per_dim = np.mean(
        np.abs(chunk_delta),
        axis=0,
    )

    # 当前 chunk 每个 action 的 L2 变化：[ACTIONS_PER_CHUNK]
    l2_delta_per_action = np.linalg.norm(
        chunk_delta,
        axis=1,
    )

    # 当前 chunk 每个 action 的平均绝对变化：[ACTIONS_PER_CHUNK]
    mean_abs_delta_per_action = np.mean(
        np.abs(chunk_delta),
        axis=1,
    )

    # 所有 chunk 的整体统计
    mean_abs_delta_all_chunks = np.mean(
        np.abs(last_delta),
        axis=(1, 2),
    )

    max_abs_delta_all_chunks = np.max(
        np.abs(last_delta),
        axis=(1, 2),
    )

    mean_l2_delta_all_chunks = np.mean(
        np.linalg.norm(last_delta, axis=2),
        axis=1,
    )

    print("\n[INFO] Final denoising transition:")
    print("       input : step=9  (x_t9)")
    print("       output: step=10 (x_t10)")
    print(f"[INFO] chunk input shape : {chunk_input.shape}")
    print(f"[INFO] chunk output shape: {chunk_output.shape}")
    print(f"[INFO] chunk delta shape : {chunk_delta.shape}")

    print(
        f"[INFO] chunk={chunk_id}, "
        f"mean_abs_delta={mean_abs_delta_chunk:.8f}, "
        f"max_abs_delta={np.max(np.abs(chunk_delta)):.8f}, "
        f"mean_l2_delta={np.mean(l2_delta_per_action):.8f}"
    )

    print("[INFO] Final-step mean absolute delta per valid dim:")
    for dim in range(VALID_DIM):
        print(
            f"    dim={dim}: "
            f"{mean_abs_delta_per_dim[dim]:.8f}"
        )

    print("[INFO] Final-step mean absolute delta per action:")
    for action_id in range(num_actions):
        print(
            f"    action={action_id}: "
            f"mean_abs={mean_abs_delta_per_action[action_id]:.8f}, "
            f"l2={l2_delta_per_action[action_id]:.8f}"
        )

    # ==============================================================
    # Figure 9a:
    # 指定 chunk 中，最后一步输入、输出和变化量
    # ==============================================================
    fig, axes = plt.subplots(
        VALID_DIM,
        1,
        figsize=(12, 2.6 * VALID_DIM),
        sharex=True,
    )

    if VALID_DIM == 1:
        axes = [axes]

    action_ids = np.arange(num_actions)

    for dim in range(VALID_DIM):
        axes[dim].plot(
            action_ids,
            chunk_input[:, dim],
            marker="o",
            linewidth=1.5,
            label="step=9 input",
        )

        axes[dim].plot(
            action_ids,
            chunk_output[:, dim],
            marker="s",
            linewidth=1.5,
            label="step=10 output",
        )

        axes[dim].plot(
            action_ids,
            chunk_delta[:, dim],
            marker="x",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label="step=10 - step=9",
        )

        axes[dim].set_ylabel(f"Dim {dim}")
        axes[dim].grid(alpha=0.3)

        if dim == 0:
            axes[dim].legend(fontsize=8, ncol=3)

    axes[-1].set_xlabel("Action index in chunk")

    fig.suptitle(
        f"Figure 9: Final Denoising Input vs Output "
        f"(Chunk {chunk_id})",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        f"fig9_final_denoising_input_vs_output_chunk{chunk_id}.png",
        dpi=150,
    )
    plt.show()
    plt.close(fig)

    # ==============================================================
    # Figure 9b:
    # 所有 chunk/action 的最后一步变化量
    # ==============================================================

    # [C, A]，每个 action 的 L2 变化
    delta_l2_all = np.linalg.norm(
        last_delta,
        axis=2,
    )

    # [C, A]，每个 action 的平均绝对变化
    mean_abs_delta_action_all = np.mean(
        np.abs(last_delta),
        axis=2,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 5),
    )

    im0 = axes[0].imshow(
        delta_l2_all,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )

    axes[0].set_title(
        "Final-step L2 change"
    )
    axes[0].set_xlabel("Action index in chunk")
    axes[0].set_ylabel("Chunk index")

    fig.colorbar(
        im0,
        ax=axes[0],
        label="L2 norm of step=10 - step=9",
    )

    im1 = axes[1].imshow(
        mean_abs_delta_action_all,
        aspect="auto",
        origin="lower",
        cmap="magma",
    )

    axes[1].set_title(
        "Final-step mean absolute change"
    )
    axes[1].set_xlabel("Action index in chunk")
    axes[1].set_ylabel("Chunk index")

    fig.colorbar(
        im1,
        ax=axes[1],
        label="Mean absolute change",
    )

    fig.suptitle(
        "Figure 9b: Final Denoising Change Across Chunks",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        "fig9b_final_denoising_change_heatmap.png",
        dpi=150,
    )
    plt.show()
    plt.close(fig)

    # ==============================================================
    # Figure 9c:
    # 当前 chunk 每个有效维度的最后一步变化量
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.bar(
        np.arange(VALID_DIM),
        mean_abs_delta_per_dim,
    )

    ax.set_xticks(np.arange(VALID_DIM))
    ax.set_xticklabels(
        [f"Dim {dim}" for dim in range(VALID_DIM)]
    )

    ax.set_xlabel("Valid action dimension")
    ax.set_ylabel("Mean absolute delta")
    ax.set_title(
        f"Figure 9c: Final-step Change per Dimension "
        f"(Chunk {chunk_id})"
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        f"fig9c_final_denoising_delta_per_dim_chunk{chunk_id}.png",
        dpi=150,
    )
    plt.show()
    plt.close(fig)

    # ==============================================================
    # Figure 9d:
    # 当前 chunk 每个 action 的变化量
    # ==============================================================

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    x = np.arange(num_actions)
    width = 0.38

    ax.bar(
        x - width / 2,
        mean_abs_delta_per_action,
        width=width,
        label="Mean absolute delta",
    )

    ax.bar(
        x + width / 2,
        l2_delta_per_action,
        width=width,
        label="L2 delta",
    )

    ax.set_xticks(x)
    ax.set_xlabel("Action index in chunk")
    ax.set_ylabel("Final-step change")
    ax.set_title(
        f"Figure 9d: Final-step Change per Action "
        f"(Chunk {chunk_id})"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        f"fig9d_final_denoising_delta_per_action_chunk{chunk_id}.png",
        dpi=150,
    )
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flow Matching denoising visualization."
        )
    )

    parser.add_argument(
        "--file",
        type=str,
        default=(
            "/opt/data/zhuoziying/wall-x/debug_output/"
            "noisy_action_trajectory.txt"
        ),
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=[
            "all",
            "overall",
            "denoising",
            "validation",
            "final_step",
        ],
        help="Which figure group to generate.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help=(
            "Chunk id for single-chunk figures. "
            f"Valid range is 0..{NUM_CHUNKS - 1}."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------
    df = load_data(args.file)

    sim_ids = get_sim_ids(df)

    if len(sim_ids) == 0:
        raise ValueError(
            "No experiment names matching 'sim_<id>' were found."
        )

    print(f"[INFO] Detected sim_ids: {sim_ids}")

    validate_data(
        df,
        sim_ids,
    )

    # --------------------------------------------------------------
    # Reconstruct tensors
    # --------------------------------------------------------------
    action_denoising = build_denoising_tensor(
        df,
        sim_ids,
    )

    final_chunks = get_final_action_chunks(
        df,
        sim_ids,
    )

    all_states = build_full_state_tensor(
        action_denoising,
        final_chunks,
    )

    print(
        f"[INFO] action_denoising shape: "
        f"{action_denoising.shape}"
    )

    print(
        f"[INFO] final_chunks shape: "
        f"{final_chunks.shape}"
    )

    print(
        f"[INFO] all_states shape: "
        f"{all_states.shape}"
    )

    expected_shape_denoising = (
        len(sim_ids),
        NUM_DENOISE_STEPS,
        ACTIONS_PER_CHUNK,
        ACTION_DIM,
    )

    expected_shape_final = (
        len(sim_ids),
        ACTIONS_PER_CHUNK,
        ACTION_DIM,
    )

    expected_shape_all = (
        len(sim_ids),
        NUM_DENOISE_STEPS + 1,
        ACTIONS_PER_CHUNK,
        ACTION_DIM,
    )

    if action_denoising.shape != expected_shape_denoising:
        raise ValueError(
            f"Unexpected action_denoising shape: "
            f"{action_denoising.shape}, "
            f"expected {expected_shape_denoising}"
        )

    if final_chunks.shape != expected_shape_final:
        raise ValueError(
            f"Unexpected final_chunks shape: "
            f"{final_chunks.shape}, "
            f"expected {expected_shape_final}"
        )

    if all_states.shape != expected_shape_all:
        raise ValueError(
            f"Unexpected all_states shape: "
            f"{all_states.shape}, "
            f"expected {expected_shape_all}"
        )

    # --------------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------------
    if args.stage in (
        "all",
        "overall",
    ):
        fig1_overall_sequence(
            final_chunks
        )

        fig2_chunk_action_heatmap(
            final_chunks
        )

    if args.stage in (
        "all",
        "denoising",
    ):
        fig3_single_chunk_denoising_lines(
            action_denoising,
            chunk_id=args.chunk,
        )

        fig4_single_chunk_denoising_heatmap(
            action_denoising,
            chunk_id=args.chunk,
        )

        fig5_mean_abs_change(
            all_states
        )

        fig6_change_heatmap(
            all_states
        )

    if args.stage in (
        "all",
        "validation",
    ):
        fig7_chunk_boundary_jump(
            final_chunks
        )

        fig8_padding_check(
            action_denoising,
            final_chunks,
        )

    if args.stage in (
        "all",
        "final_step",
        "validation",
    ):
        fig9_final_denoising_input_vs_output(
            all_states,
            chunk_id=args.chunk,
        )

    print(
        "[INFO] Done. Figures saved."
    )


if __name__ == "__main__":
    main()
