import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ----- File Locations -----
RAW_FILE = "Data_collection/variances/output_wrench_log_test.csv"
KF_FILE  = "Data_collection/filtered/Filtered_wrench_output.csv"

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

def plot_single_comparison(raw_df, kf_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Fz comparison ---
    axes[0].plot(raw_df["seq"], raw_df["Fz"], label="Raw", alpha=0.5)
    axes[0].plot(kf_df["time"], kf_df["fz"], label="KF Filtered")
    axes[0].set_title("Fz [N]")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Force Z [N]")
    axes[0].grid(True)
    axes[0].legend()

    # --- Ty comparison ---
    axes[1].plot(raw_df["seq"], raw_df["My"], label="Raw", alpha=0.5)
    axes[1].plot(kf_df["time"], kf_df["ty"], label="KF Filtered")
    axes[1].set_title("Ty [Nm]")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Torque Y [Nm]")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw_df = load_raw_wrench(RAW_FILE)
    kf_df  = load_kf(KF_FILE)

    if raw_df is not None and kf_df is not None:
        plot_single_comparison(raw_df, kf_df)
    else:
        print("❌ Missing required data files")
