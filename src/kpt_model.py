import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os

#Load datasets
dirty_df = pd.read_csv("data/synthetic_orders.csv")
clean_df = pd.read_csv("data/enriched_orders.csv")

print(f"Dirty dataset: {len(dirty_df)} orders")
print(f"Clean/enriched dataset: {len(clean_df)} orders")

#Feature columns
base_features = [
    "concurrent_orders",
    "is_peak_hour",
    "mx_response_latency_sec",
    "rejection_flag"
]

enriched_features = base_features + ["computed_kls"]

# Encode categorical for dirty dataset
dirty_df["restaurant_type_enc"] = dirty_df["restaurant_type"].map(
    {"fast_food": 0, "casual_dining": 1, "cloud_kitchen": 2}
)
dirty_df["restaurant_size_enc"] = dirty_df["restaurant_size"].map(
    {"small": 0, "medium": 1, "large": 2}
)

clean_df["restaurant_type_enc"] = clean_df["restaurant_type"].map(
    {"fast_food": 0, "casual_dining": 1, "cloud_kitchen": 2}
)
clean_df["restaurant_size_enc"] = clean_df["restaurant_size"].map(
    {"small": 0, "medium": 1, "large": 2}
)

base_features += ["restaurant_type_enc", "restaurant_size_enc"]
enriched_features += ["restaurant_type_enc", "restaurant_size_enc"]

#Baseline Model (dirty labels, no KLS)
X_dirty = dirty_df[base_features]
y_dirty = dirty_df["recorded_kpt_minutes"]  # biased labels
y_dirty_true = dirty_df["true_kpt_minutes"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dirty, y_dirty, test_size=0.2, random_state=42
)
_, _, _, y_test_true_d = train_test_split(
    X_dirty, y_dirty_true, test_size=0.2, random_state=42
)

baseline_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
baseline_model.fit(X_train_d, y_train_d)
baseline_preds = baseline_model.predict(X_test_d)
baseline_mae = mean_absolute_error(y_test_true_d, baseline_preds)

print(f"\n--- Baseline Model (dirty labels, no Kitchen Load Score) ---")
print(f"MAE vs True KPT: {baseline_mae:.2f} minutes")

#Enriched Model (clean labels + KLS)
X_clean = clean_df[enriched_features]
y_clean = clean_df["recorded_kpt_minutes"]  # clean labels
y_clean_true = clean_df["true_kpt_minutes"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)
_, _, _, y_test_true_c = train_test_split(
    X_clean, y_clean_true, test_size=0.2, random_state=42
)

enriched_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
enriched_model.fit(X_train_c, y_train_c)
enriched_preds = enriched_model.predict(X_test_c)
enriched_mae = mean_absolute_error(y_test_true_c, enriched_preds)

print(f"\n--- Enriched Model (clean labels + Kitchen Load Score) ---")
print(f"MAE vs True KPT: {enriched_mae:.2f} minutes")

#Improvement Summary
improvement = baseline_mae - enriched_mae
improvement_pct = (improvement / baseline_mae) * 100

print(f"\n--- Improvement Summary ---")
print(f"Baseline MAE:   {baseline_mae:.2f} mins")
print(f"Enriched MAE:   {enriched_mae:.2f} mins")
print(f"Improvement:    {improvement:.2f} mins ({improvement_pct:.1f}% reduction in error)")

#Visualizations
os.makedirs("outputs", exist_ok=True)

# Plot 1: MAE Comparison Bar Chart
plt.figure(figsize=(7, 5))
models = ["Baseline\n(Dirty Labels)", "Enriched\n(Clean Labels + KLS)"]
maes = [baseline_mae, enriched_mae]
colors = ["salmon", "steelblue"]
bars = plt.bar(models, maes, color=colors, edgecolor="white", width=0.4)
for bar, mae in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f"{mae:.2f} min", ha="center", va="bottom", fontweight="bold")
plt.ylabel("Mean Absolute Error (minutes)")
plt.title(f"KPT Prediction MAE: Baseline vs Enriched\n({improvement_pct:.1f}% improvement)")
plt.ylim(0, max(maes) * 1.3)
plt.tight_layout()
plt.savefig("outputs/mae_comparison.png", dpi=150)
plt.close()
print("\nChart saved: outputs/mae_comparison.png")

# Plot 2: Feature Importance
plt.figure(figsize=(8, 5))
importance = pd.Series(enriched_model.feature_importances_, index=enriched_features)
importance.sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
plt.xlabel("Feature Importance")
plt.title("Enriched Model — Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.close()
print("Chart saved: outputs/feature_importance.png")

# Plot 3: Predicted vs Actual KPT
plt.figure(figsize=(8, 5))
plt.scatter(y_test_true_c, enriched_preds, alpha=0.3, color="steelblue", s=10)
plt.plot([5, 60], [5, 60], "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("True KPT (minutes)")
plt.ylabel("Predicted KPT (minutes)")
plt.title("Enriched Model: Predicted vs True KPT")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/predicted_vs_actual.png", dpi=150)
plt.close()
print("Chart saved: outputs/predicted_vs_actual.png")

print("\nKPT Model training complete.")

