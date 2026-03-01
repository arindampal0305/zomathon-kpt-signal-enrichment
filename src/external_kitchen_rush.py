import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

#Configuration
NUM_RESTAURANTS = 50
HOURS = list(range(0, 24))
RESTAURANT_IDS = [f"REST_{i:03d}" for i in range(1, NUM_RESTAURANTS + 1)]

#Simulate hourly order volumes per restaurant
records = []

for restaurant_id in RESTAURANT_IDS:
    restaurant_size = np.random.choice(["small", "medium", "large"], p=[0.4, 0.4, 0.2])
    base_capacity = {"small": 5, "medium": 12, "large": 25}[restaurant_size]

    for hour in HOURS:
        is_peak = hour in [12, 13, 14, 19, 20, 21]

        # Zomato orders (what the system currently sees)
        zomato_orders = max(0, int(np.random.poisson(
            base_capacity * 0.4 * (1.8 if is_peak else 1.0)
        )))

        # Dine-in orders (invisible to Zomato)
        dine_in_orders = max(0, int(np.random.poisson(
            base_capacity * 0.35 * (2.0 if is_peak else 0.8)
        )))

        # Competitor platform orders (Swiggy, etc.)  invisible to Zomato
        competitor_orders = max(0, int(np.random.poisson(
            base_capacity * 0.25 * (1.5 if is_peak else 0.9)
        )))

        total_orders = zomato_orders + dine_in_orders + competitor_orders

        # KPT underestimation when ignoring external load
        # True KPT accounts for total kitchen load
        true_kpt = max(5, round(
            10 + total_orders * 1.1 + (5 if is_peak else 0) + np.random.normal(0, 2), 1
        ))

        # Zomato-only KPT (what current model estimates, ignores external load)
        zomato_only_kpt = max(5, round(
            10 + zomato_orders * 1.1 + (5 if is_peak else 0) + np.random.normal(0, 2), 1
        ))

        underestimation = true_kpt - zomato_only_kpt

        records.append({
            "restaurant_id": restaurant_id,
            "restaurant_size": restaurant_size,
            "hour": hour,
            "is_peak": int(is_peak),
            "zomato_orders": zomato_orders,
            "dine_in_orders": dine_in_orders,
            "competitor_orders": competitor_orders,
            "total_orders": total_orders,
            "true_kpt": true_kpt,
            "zomato_only_kpt": zomato_only_kpt,
            "kpt_underestimation": underestimation
        })

df = pd.DataFrame(records)

#Summary Stats
print("--- External Kitchen Rush Analysis ---")
print(f"Avg Zomato-only orders per hour:     {df['zomato_orders'].mean():.1f}")
print(f"Avg dine-in orders per hour:         {df['dine_in_orders'].mean():.1f}")
print(f"Avg competitor orders per hour:      {df['competitor_orders'].mean():.1f}")
print(f"Avg total orders per hour:           {df['total_orders'].mean():.1f}")
print(f"\nAvg KPT (Zomato-only estimate):      {df['zomato_only_kpt'].mean():.2f} mins")
print(f"Avg KPT (true, with external load):  {df['true_kpt'].mean():.2f} mins")
print(f"Avg KPT underestimation:             {df['kpt_underestimation'].mean():.2f} mins")
print(f"Peak hour underestimation:           {df[df['is_peak']==1]['kpt_underestimation'].mean():.2f} mins")
print(f"Off-peak underestimation:            {df[df['is_peak']==0]['kpt_underestimation'].mean():.2f} mins")

#Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/external_kitchen_rush.csv", index=False)
print(f"\nDataset saved: data/external_kitchen_rush.csv")

#Visualizations
os.makedirs("outputs", exist_ok=True)

# Plot 1: Order breakdown by hour (stacked bar)
hourly = df.groupby("hour")[["zomato_orders", "dine_in_orders", "competitor_orders"]].mean()
plt.figure(figsize=(12, 5))
plt.bar(hourly.index, hourly["zomato_orders"], label="Zomato Orders", color="steelblue")
plt.bar(hourly.index, hourly["dine_in_orders"], bottom=hourly["zomato_orders"],
        label="Dine-in Orders", color="coral")
plt.bar(hourly.index, hourly["competitor_orders"],
        bottom=hourly["zomato_orders"] + hourly["dine_in_orders"],
        label="Competitor Orders", color="gold")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Orders")
plt.title("Kitchen Load Breakdown by Hour\n(Zomato sees only the blue portion)")
plt.legend()
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("outputs/kitchen_load_breakdown.png", dpi=150)
plt.close()
print("Chart saved: outputs/kitchen_load_breakdown.png")

# Plot 2: KPT underestimation by hour
hourly_kpt = df.groupby("hour")[["true_kpt", "zomato_only_kpt"]].mean()
plt.figure(figsize=(12, 5))
plt.plot(hourly_kpt.index, hourly_kpt["true_kpt"], color="steelblue",
         linewidth=2, marker="o", label="True KPT (with external load)")
plt.plot(hourly_kpt.index, hourly_kpt["zomato_only_kpt"], color="salmon",
         linewidth=2, marker="o", linestyle="--", label="Zomato-only KPT estimate")
plt.fill_between(hourly_kpt.index, hourly_kpt["zomato_only_kpt"],
                 hourly_kpt["true_kpt"], alpha=0.2, color="red", label="Underestimation gap")
plt.xlabel("Hour of Day")
plt.ylabel("KPT (minutes)")
plt.title("KPT Underestimation Due to Invisible Kitchen Load\n(Gap = missed dine-in + competitor orders)")
plt.legend()
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("outputs/kpt_underestimation_by_hour.png", dpi=150)
plt.close()
print("Chart saved: outputs/kpt_underestimation_by_hour.png")

# Plot 3: Underestimation by restaurant size
plt.figure(figsize=(7, 5))
size_under = df.groupby("restaurant_size")["kpt_underestimation"].mean()
colors = {"small": "steelblue", "medium": "coral", "large": "gold"}
bars = plt.bar(size_under.index, size_under.values,
               color=[colors[s] for s in size_under.index], edgecolor="white")
for bar, val in zip(bars, size_under.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f} min", ha="center", fontweight="bold")
plt.ylabel("Avg KPT Underestimation (minutes)")
plt.title("KPT Underestimation by Restaurant Size\n(Larger restaurants have more invisible load)")
plt.tight_layout()
plt.savefig("outputs/underestimation_by_size.png", dpi=150)
plt.close()
print("Chart saved: outputs/underestimation_by_size.png")

print("\nExternal Kitchen Rush analysis complete.")
