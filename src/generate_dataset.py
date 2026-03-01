import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

# --- Config ---
NUM_RESTAURANTS = 50
NUM_ORDERS = 5000
CONTAMINATION_RATE = 0.45  # 45% of FOR signals are rider-influenced

# --- Generate Restaurants ---
restaurant_ids = [f"REST_{i:03d}" for i in range(1, NUM_RESTAURANTS + 1)]
restaurant_types = {r: random.choice(["fast_food", "casual_dining", "cloud_kitchen"]) for r in restaurant_ids}
restaurant_sizes = {r: random.choice(["small", "medium", "large"]) for r in restaurant_ids}

# --- Generate Orders ---
records = []
base_date = datetime(2024, 1, 1)

for order_num in range(NUM_ORDERS):
    order_id = f"ORD_{order_num:05d}"
    restaurant_id = random.choice(restaurant_ids)
    r_type = restaurant_types[restaurant_id]
    r_size = restaurant_sizes[restaurant_id]

    # Order placed timestamp
    order_time = base_date + timedelta(
        days=random.randint(0, 180),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    # Hour of day (for peak hours)
    hour = order_time.hour
    is_peak = 1 if hour in [12, 13, 14, 19, 20, 21] else 0

    # Concurrent orders at this restaurant (higher during peak)
    base_concurrent = {"small": 2, "medium": 5, "large": 10}[r_size]
    concurrent_orders = max(1, int(np.random.poisson(base_concurrent * (1.8 if is_peak else 1.0))))

    # Kitchen Load Score (0-1) — derived from concurrent orders + peak flag
    kitchen_load_score = min(1.0, round((concurrent_orders / 20) + (0.2 if is_peak else 0), 2))

    # Mx app response latency (seconds) — higher when kitchen is busy
    mx_response_latency = round(np.random.normal(3 + kitchen_load_score * 5, 1.5), 2)
    mx_response_latency = max(0.5, mx_response_latency)

    # Order rejection flag — more likely when overloaded
    rejection_flag = 1 if random.random() < (0.05 + kitchen_load_score * 0.2) else 0

    # True KPT (minutes) — what we want to predict
    base_kpt = {"fast_food": 12, "casual_dining": 22, "cloud_kitchen": 18}[r_type]
    true_kpt = max(5, round(np.random.normal(
        base_kpt + concurrent_orders * 1.2 + (5 if is_peak else 0),
        4
    ), 1))

    # Rider dispatched at order_time + (true_kpt - some offset)
    rider_dispatch_offset = max(1, true_kpt - random.randint(2, 6))
    rider_arrival_at_restaurant = order_time + timedelta(minutes=rider_dispatch_offset + random.randint(3, 8))

    # FOR timestamp — contaminated or clean
    is_contaminated = 1 if random.random() < CONTAMINATION_RATE else 0

    if is_contaminated:
        # Merchant taps FOR within 60 seconds of rider arrival (biased signal)
        for_delta_seconds = random.randint(0, 59)
        for_timestamp = rider_arrival_at_restaurant + timedelta(seconds=for_delta_seconds)
        recorded_kpt = (for_timestamp - order_time).total_seconds() / 60
        recorded_kpt = recorded_kpt * random.uniform(1.1, 1.4)  # amplify bias
    else:
        # Clean signal — FOR tapped when food is actually ready
        for_timestamp = order_time + timedelta(minutes=true_kpt + random.randint(-1, 2))
        recorded_kpt = (for_timestamp - order_time).total_seconds() / 60

    recorded_kpt = max(1, round(recorded_kpt, 1))

    records.append({
        "order_id": order_id,
        "restaurant_id": restaurant_id,
        "restaurant_type": r_type,
        "restaurant_size": r_size,
        "order_timestamp": order_time,
        "for_timestamp": for_timestamp,
        "rider_arrival_timestamp": rider_arrival_at_restaurant,
        "true_kpt_minutes": true_kpt,
        "recorded_kpt_minutes": recorded_kpt,
        "concurrent_orders": concurrent_orders,
        "kitchen_load_score": kitchen_load_score,
        "mx_response_latency_sec": mx_response_latency,
        "rejection_flag": rejection_flag,
        "is_peak_hour": is_peak,
        "is_contaminated": is_contaminated  # ground truth label for evaluation
    })

df = pd.DataFrame(records)

# Save to data folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_orders.csv", index=False)

print(f"Dataset generated: {len(df)} orders across {NUM_RESTAURANTS} restaurants")
print(f"Contamination rate: {df['is_contaminated'].mean():.1%}")
print(f"Avg true KPT: {df['true_kpt_minutes'].mean():.1f} mins")
print(f"Avg recorded KPT: {df['recorded_kpt_minutes'].mean():.1f} mins")
print("\nSample rows:")
print(df.head(3).to_string())
