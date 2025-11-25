import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ----- File Locations -----
RAW_FILE = "Data_collection/variances/output_wrench_log_test_standing_still.csv"
KF_FILE  = "Data_collection/filtered/Filtered_wrench_output_test_standing_still.csv"

def load_raw_wrench(file):
    """Load raw wrench CSV: seq,Fx,Fy,Fz,Mx,My,Mz,s0..."""
    if not os.path.exists(file):
        print(f"❌ Raw file not found: {file}")
        return None
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def load_kf(file):
    """Load KF filtered CSV: time,a_x,a_y,a_z,F_x...T_z"""
    if not os.path.exists(file):
        print(f"❌ KF file not found: {file}")
        return None
    df = pd.read_csv(file)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def plot_all_axes(raw_df, kf_df):
    """Plot all forces (Fx,Fy,Fz) and torques (Tx,Ty,Tz) raw vs filtered."""

    force_axes  = [("Fx", "fx"), ("Fy", "fy"), ("Fz", "fz")]
    torque_axes = [("Mx", "tx"), ("My", "ty"), ("Mz", "tz")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # ----- Plot Forces -----
    for i, (raw_key, kf_key) in enumerate(force_axes):
        ax = axes[0, i]
        ax.plot(raw_df["time"], raw_df[raw_key], label="Raw", alpha=0.5)
        ax.plot(kf_df["time"],  kf_df[kf_key],  label="KF Filtered")
        ax.set_title(f"{raw_key} [N]")
        ax.set_xlabel("Time")
        ax.set_ylabel("Force [N]")
        ax.grid(True)
        ax.legend()

    # ----- Plot Torques -----
    for i, (raw_key, kf_key) in enumerate(torque_axes):
        ax = axes[1, i]
        ax.plot(raw_df["time"], raw_df[raw_key], label="Raw", alpha=0.5)
        ax.plot(kf_df["time"],  kf_df[kf_key],  label="KF Filtered")
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
