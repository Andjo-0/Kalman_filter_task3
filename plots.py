import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

# ---- Settings ----
RAW_DIR = "Filtered_data"        # directory with raw CSVs
KF_DIR = "Filtered_data"         # directory with KF CSVs
EXPERIMENTS = [
    "1-baseline",
    "2-vibrations",
    "3-vibrations-contact"
]

def load_csv_data(directory, tag, raw=True):
    """Load CSV and return DataFrame with proper columns"""
    if raw:
        filename = f"{tag}_accel_raw.csv"  # or wrench_raw.csv depending on type
        # We'll pick wrench columns here for consistency
        filename = f"{tag}_wrench_raw.csv"
    else:
        filename = f"{tag}_kf.csv"

    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        return None

    df = pd.read_csv(path)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

def plot_comparison(experiments):
    fig, axes = plt.subplots(len(experiments), 2, figsize=(12, 4*len(experiments)))
    if len(experiments) == 1:
        axes = axes.reshape(1,2)

    for i, tag in enumerate(experiments):
        raw_df = load_csv_data(RAW_DIR, tag, raw=True)
        kf_df = load_csv_data(KF_DIR, tag, raw=False)
        if raw_df is None or kf_df is None:
            continue

        # --- Fz comparison ---
        axes[i,0].plot(raw_df["time"], raw_df["fz"], label="Raw", color='blue', alpha=0.5)
        axes[i,0].plot(kf_df["time"], kf_df["F_z"], label="KF Filtered", color='red')
        axes[i,0].set_title(f"{tag} - Fz [N]")
        axes[i,0].set_xlabel("Time [s]")
        axes[i,0].set_ylabel("Force Z [N]")
        axes[i,0].grid(True)
        axes[i,0].legend()

        # --- Ty comparison ---
        axes[i,1].plot(raw_df["time"], raw_df["ty"], label="Raw", color='blue', alpha=0.5)
        axes[i,1].plot(kf_df["time"], kf_df["T_y"], label="KF Filtered", color='red')
        axes[i,1].set_title(f"{tag} - Ty [Nm]")
        axes[i,1].set_xlabel("Time [s]")
        axes[i,1].set_ylabel("Torque Y [Nm]")
        axes[i,1].grid(True)
        axes[i,1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    available_experiments = []
    for tag in EXPERIMENTS:
        raw_path = os.path.join(RAW_DIR, f"{tag}_wrench_raw.csv")
        kf_path = os.path.join(KF_DIR, f"{tag}_kf.csv")
        if os.path.exists(raw_path) and os.path.exists(kf_path):
            available_experiments.append(tag)
        else:
            print(f"⚠️ Missing data for {tag}, skipping...")

    if available_experiments:
        plot_comparison(available_experiments)
