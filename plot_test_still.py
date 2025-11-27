import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ----- File Locations -----
RAW_FILE = "Data_collection/variances/output_wrench_log_test_standing_still.csv"
KF_FILE  = "Data_collection/filtered/Filtered_wrench_output_test_standing_still.csv"

def load_raw_wrench(file):
    if not os.path.exists(file):
        print(f"❌ Raw file not found: {file}")
        return None
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def load_kf(file):
    if not os.path.exists(file):
        print(f"❌ KF file not found: {file}")
        return None
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def plot_all_axes(raw_df, kf_df):
    """Plot forces and torques raw vs filtered vs KF state components."""

    force_axes  = [("Fx", "F_x", "z_1"),
                   ("Fy", "F_y", "z_2"),
                   ("Fz", "F_z", "z_3")]

    torque_axes = [("Mx", "T_x", "z_4"),
                   ("My", "T_y", "z_5"),
                   ("Mz", "T_z", "z_6")]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # ----- Forces -----
    for i, (raw_key, kf_key, kf_extra) in enumerate(force_axes):
        ax = axes[0, i]
        ax.plot(raw_df["time"], raw_df[raw_key], label="Raw", alpha=0.5)
        ax.plot(kf_df["time"],  kf_df[kf_key],  label="KF Filtered")
        ax.plot(kf_df["time"],  kf_df[kf_extra], label=f"KF {kf_extra}", color="green")
        ax.set_title(f"{raw_key} [N]")
        ax.set_xlabel("Time")
        ax.set_ylabel("Force [N]")
        ax.grid(True)
        ax.legend()

    # ----- Torques -----
    for i, (raw_key, kf_key, kf_extra) in enumerate(torque_axes):
        ax = axes[1, i]
        ax.plot(raw_df["time"], raw_df[raw_key], label="Raw", alpha=0.5)
        ax.plot(kf_df["time"],  kf_df[kf_key],  label="KF Filtered")
        ax.plot(kf_df["time"],  kf_df[kf_extra], label=f"KF {kf_extra}", color="green")
        ax.set_title(f"{raw_key} [Nm]")
        ax.set_xlabel("Time")
        ax.set_ylabel("Torque [Nm]")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw_df = load_raw_wrench(RAW_FILE)
    kf_df  = load_kf(KF_FILE)

    if raw_df is not None and kf_df is not None:
        plot_all_axes(raw_df, kf_df)
    else:
        print("❌ Missing required data files")
