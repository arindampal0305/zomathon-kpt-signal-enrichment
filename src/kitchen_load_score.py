import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Load clean dataset
df = pd.read_csv("data/clean_orders.csv", parse_dates=[
    "order_timestamp", "for_timestamp", "rider_arrival_timestamp"
])

print(f"Orders loaded: {len(df)}")

#Extract hour and date for grouping
df["order_hour"] = df["order_timestamp"].dt.hour
df["order_date"] = df["order_timestamp"].dt.date

#Per-restaurant rolling Kitchen Load Score
#We'll compute it per restaurant per hour window

restaurant_stats = []

for restaurant_id, group in df.groupby("restaurant_id"):
    group = group.sort_values("order_timestamp").copy()

    # Rolling 3-order window stats
    group["rolling_avg_latency"] = group["mx_response_latency_sec"].rolling(3, min_periods=1).mean()
    group["rolling_rejection_rate"] = group["rejection_flag"].rolling(3, min_periods=1).mean()
    group["rolling_concurrent"] = group["concurrent_orders"].rolling(3, min_periods=1).mean()

    # Normalize each component to 0-1
    max_latency = df["mx_response_latency_sec"].max()
    max_concurrent = df["concurrent_orders"].max()

    group["latency_score"] = group["rolling_avg_latency"] / max_latency
    group["concurrent_score"] = group["rolling_concurrent"] / max_concurrent
    group["rejection_score"] = group["rolling_rejection_rate"]  # already 0-1

    # Weighted Kitchen Load Score
    group["computed_kls"] = (
        0.4 * group["latency_score"] +
        0.4 * group["concurrent_score"] +
        0.2 * group["rejection_score"]
    ).round(3)

    restaurant_stats.append(group)

df_enriched = pd.concat(restaurant_stats).sort_index()

print(f"\n--- Kitchen Load Score Stats ---")
print(f"Avg Kitchen Load Score:  {df_enriched['computed_kls'].mean():.3f}")
print(f"Max Kitchen Load Score:  {df_enriched['computed_kls'].max():.3f}")
print(f"Min Kitchen Load Score:  {df_enriched['computed_kls'].min():.3f}")

#Validate: does higher KLS correlate with higher true KPT?
correlation = df_enriched["computed_kls"].corr(df_enriched["true_kpt_minutes"])
print(f"\nCorrelation between Kitchen Load Score and True KPT: {correlation:.3f}")
print("(Positive correlation validates that KLS captures real kitchen stress)")

#Save enriched dataset
df_enriched.to_csv("data/enriched_orders.csv", index=False)
print(f"\nEnriched dataset saved: data/enriched_orders.csv")

#Visualizations
os.makedirs("outputs", exist_ok=True)

# Plot 1: Kitchen Load Score vs True KPT scatter
plt.figure(figsize=(8, 5))
plt.scatter(df_enriched["computed_kls"], df_enriched["true_kpt_minutes"],
            alpha=0.3, color="steelblue", s=10)
plt.xlabel("Kitchen Load Score (0-1)")
plt.ylabel("True KPT (minutes)")
plt.title(f"Kitchen Load Score vs True KPT\n(Correlation: {correlation:.3f})")
plt.tight_layout()
plt.savefig("outputs/kls_vs_true_kpt.png", dpi=150)
plt.close()
print("Chart saved: outputs/kls_vs_true_kpt.png")

# Plot 2: Average KLS by hour of day
hourly_kls = df_enriched.groupby("order_hour")["computed_kls"].mean()
plt.figure(figsize=(10, 4))
plt.bar(hourly_kls.index, hourly_kls.values, color="coral", edgecolor="white")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Kitchen Load Score")
plt.title("Kitchen Load Score by Hour of Day\n(Peak hours should show higher load)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("outputs/kls_by_hour.png", dpi=150)
plt.close()
print("Chart saved: outputs/kls_by_hour.png")

# Plot 3: KLS distribution by restaurant size
plt.figure(figsize=(8, 5))
for size, color in [("small", "steelblue"), ("medium", "coral"), ("large", "green")]:
    subset = df_enriched[df_enriched["restaurant_size"] == size]["computed_kls"]
    plt.hist(subset, bins=30, alpha=0.6, label=size, color=color, edgecolor="white")
plt.xlabel("Kitchen Load Score")
plt.ylabel("Count")
plt.title("Kitchen Load Score Distribution by Restaurant Size")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/kls_by_restaurant_size.png", dpi=150)
plt.close()
print("Chart saved: outputs/kls_by_restaurant_size.png")

print("\nKitchen Load Score computation complete.")
