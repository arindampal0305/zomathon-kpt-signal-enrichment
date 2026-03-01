import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Dataset ---
df = pd.read_csv("data/synthetic_orders.csv", parse_dates=[
    "order_timestamp", "for_timestamp", "rider_arrival_timestamp"
])

print(f"Total orders loaded: {len(df)}")

# --- Core Logic: Compute delta between FOR tap and rider arrival ---
df["for_rider_delta_seconds"] = (
    df["for_timestamp"] - df["rider_arrival_timestamp"]
).dt.total_seconds()

# --- Flag rider-influenced FOR signals ---
DELTA_THRESHOLD = 60  # seconds

df["is_rider_influenced"] = (
    df["for_rider_delta_seconds"].abs() <= DELTA_THRESHOLD
).astype(int)

# --- Split into clean and contaminated ---
clean_df = df[df["is_rider_influenced"] == 0].copy()
contaminated_df = df[df["is_rider_influenced"] == 1].copy()

print(f"\n--- Signal Validation Results ---")
print(f"Total orders:            {len(df)}")
print(f"Rider-influenced (noisy): {len(contaminated_df)} ({len(contaminated_df)/len(df):.1%})")
print(f"Clean FOR signals:        {len(clean_df)} ({len(clean_df)/len(df):.1%})")

# --- Show KPT bias introduced by contaminated labels ---
print(f"\n--- KPT Label Bias Analysis ---")
print(f"Avg recorded KPT (all data):       {df['recorded_kpt_minutes'].mean():.2f} mins")
print(f"Avg recorded KPT (contaminated):   {contaminated_df['recorded_kpt_minutes'].mean():.2f} mins")
print(f"Avg recorded KPT (clean):          {clean_df['recorded_kpt_minutes'].mean():.2f} mins")
print(f"Avg TRUE KPT (all data):           {df['true_kpt_minutes'].mean():.2f} mins")
print(f"Bias introduced by contamination:  {df['recorded_kpt_minutes'].mean() - df['true_kpt_minutes'].mean():.2f} mins")

# --- Save clean dataset ---
os.makedirs("data", exist_ok=True)
clean_df.to_csv("data/clean_orders.csv", index=False)
print(f"\nClean dataset saved: data/clean_orders.csv ({len(clean_df)} orders)")

# --- Visualizations ---
os.makedirs("outputs", exist_ok=True)

# Plot 1: Distribution of FOR-Rider delta
plt.figure(figsize=(10, 5))
plt.hist(df["for_rider_delta_seconds"], bins=100, color="steelblue", edgecolor="white")
plt.axvline(x=60, color="red", linestyle="--", linewidth=2, label="60s threshold")
plt.axvline(x=-60, color="red", linestyle="--", linewidth=2)
plt.xlabel("FOR Timestamp - Rider Arrival (seconds)")
plt.ylabel("Number of Orders")
plt.title("Distribution of FOR-Rider Delta\n(Spike near 0 = rider-influenced signals)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/for_rider_delta_distribution.png", dpi=150)
plt.close()
print("Chart saved: outputs/for_rider_delta_distribution.png")

# Plot 2: Recorded KPT vs True KPT — clean vs contaminated
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(contaminated_df["recorded_kpt_minutes"], bins=40, color="salmon", edgecolor="white", alpha=0.8)
axes[0].hist(contaminated_df["true_kpt_minutes"], bins=40, color="steelblue", edgecolor="white", alpha=0.6)
axes[0].set_title("Contaminated Labels\nRecorded KPT vs True KPT")
axes[0].set_xlabel("KPT (minutes)")
axes[0].legend(["Recorded KPT", "True KPT"])

axes[1].hist(clean_df["recorded_kpt_minutes"], bins=40, color="salmon", edgecolor="white", alpha=0.8)
axes[1].hist(clean_df["true_kpt_minutes"], bins=40, color="steelblue", edgecolor="white", alpha=0.6)
axes[1].set_title("Clean Labels\nRecorded KPT vs True KPT")
axes[1].set_xlabel("KPT (minutes)")
axes[1].legend(["Recorded KPT", "True KPT"])

plt.suptitle("Label Quality: Contaminated vs Clean FOR Signals", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/label_quality_comparison.png", dpi=150)
plt.close()
print("Chart saved: outputs/label_quality_comparison.png")

print("\nSignal Validator complete.")
