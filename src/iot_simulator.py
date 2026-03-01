import time
import random
import json
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import os

#IoT Simulator
# Simulates an ESP32 sensor at the kitchen pass that detects
# when a food bag is placed on the counter (bag ready event)
# In reality this would use MQTT protocol to send events
# Here we simulate the event stream and show label quality improvement

random.seed(42)

#Config
NUM_EVENTS = 200
RESTAURANT_ID = "REST_001"  # Simulating a single high-volume restaurant

print("=" * 55)
print("  IoT Sensor Simulator — Kitchen Pass Detection")
print("  Simulating ESP32 sensor at kitchen pass")
print("=" * 55)
print(f"\nRestaurant: {RESTAURANT_ID}")
print("Sensor type: Acoustic + Weight (bag placement detection)")
print("Protocol: MQTT (simulated)")
print("Broker: Mosquitto (simulated)\n")
print("--- Live Event Stream (last 10 shown) ---\n")

#Simulate event stream
events = []
base_time = datetime(2024, 6, 1, 10, 0, 0)

for i in range(NUM_EVENTS):
    order_id = f"ORD_{i:05d}"

    #Order placed
    order_time = base_time + timedelta(minutes=i * 3 + random.randint(0, 2))

    #True food ready time (what sensor detects)
    true_kpt = max(5, round(random.gauss(20, 5), 1))
    bag_ready_time = order_time + timedelta(minutes=true_kpt)

    #What merchant would have tapped (biased: rider influenced)
    rider_arrival = order_time + timedelta(minutes=true_kpt - random.randint(2, 8))
    is_contaminated = random.random() < 0.45
    if is_contaminated:
        merchant_for_time = rider_arrival + timedelta(seconds=random.randint(0, 59))
    else:
        merchant_for_time = order_time + timedelta(minutes=true_kpt + random.randint(-1, 2))

    merchant_kpt = (merchant_for_time - order_time).total_seconds() / 60
    iot_kpt = (bag_ready_time - order_time).total_seconds() / 60

    #Simulate MQTT event payload
    mqtt_payload = {
        "event_type": "BAG_READY",
        "sensor_id": f"SENSOR_{RESTAURANT_ID}",
        "restaurant_id": RESTAURANT_ID,
        "order_id": order_id,
        "timestamp": bag_ready_time.strftime("%Y-%m-%d %H:%M:%S"),
        "iot_kpt_minutes": round(iot_kpt, 1),
        "confidence": round(random.uniform(0.92, 0.99), 2)
    }

    events.append({
        "order_id": order_id,
        "order_timestamp": order_time,
        "bag_ready_timestamp": bag_ready_time,
        "merchant_for_timestamp": merchant_for_time,
        "true_kpt_minutes": true_kpt,
        "iot_kpt_minutes": round(iot_kpt, 1),
        "merchant_kpt_minutes": round(merchant_kpt, 1),
        "is_contaminated": int(is_contaminated),
        "mqtt_payload": json.dumps(mqtt_payload)
    })

    # Print last 10 events as live stream
    if i >= NUM_EVENTS - 10:
        print(f"[{bag_ready_time.strftime('%H:%M:%S')}] "
              f"ORDER {order_id} | "
              f"IoT KPT: {round(iot_kpt,1)} min | "
              f"Merchant KPT: {round(merchant_kpt,1)} min | "
              f"{'⚠ BIASED' if is_contaminated else '✓ CLEAN'}")

df = pd.DataFrame(events)

#Label Quality Comparison
print("\n--- IoT Label Quality Analysis ---")
print(f"Total events captured:         {len(df)}")
print(f"Contaminated merchant signals: {df['is_contaminated'].sum()} ({df['is_contaminated'].mean():.1%})")
print(f"\nAvg IoT KPT:                   {df['iot_kpt_minutes'].mean():.2f} mins")
print(f"Avg Merchant KPT:              {df['merchant_kpt_minutes'].mean():.2f} mins")
print(f"Avg True KPT:                  {df['true_kpt_minutes'].mean():.2f} mins")
print(f"\nIoT label error vs true KPT:       {abs(df['iot_kpt_minutes'].mean() - df['true_kpt_minutes'].mean()):.2f} mins")
print(f"Merchant label error vs true KPT:  {abs(df['merchant_kpt_minutes'].mean() - df['true_kpt_minutes'].mean()):.2f} mins")
print(f"\nIoT sensor provides {abs(df['merchant_kpt_minutes'].mean() - df['true_kpt_minutes'].mean()):.2f}x cleaner labels than merchant taps")

#Save
os.makedirs("data", exist_ok=True)
df.to_csv("data/iot_events.csv", index=False)
print(f"\nIoT events saved: data/iot_events.csv")

#Visualizations
os.makedirs("outputs", exist_ok=True)

# Plot 1: IoT vs Merchant KPT vs True KPT
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["true_kpt_minutes"], color="steelblue",
         alpha=0.6, linewidth=1, label="True KPT")
plt.plot(df.index, df["iot_kpt_minutes"], color="green",
         alpha=0.8, linewidth=1, linestyle="--", label="IoT Sensor KPT")
plt.plot(df.index, df["merchant_kpt_minutes"], color="salmon",
         alpha=0.6, linewidth=1, linestyle=":", label="Merchant FOR KPT")
plt.xlabel("Order Index")
plt.ylabel("KPT (minutes)")
plt.title("Label Quality: IoT Sensor vs Merchant FOR vs True KPT")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/iot_label_quality.png", dpi=150)
plt.close()
print("Chart saved: outputs/iot_label_quality.png")

# Plot 2: Error distribution comparison
plt.figure(figsize=(9, 5))
iot_errors = (df["iot_kpt_minutes"] - df["true_kpt_minutes"]).abs()
merchant_errors = (df["merchant_kpt_minutes"] - df["true_kpt_minutes"]).abs()
plt.hist(merchant_errors, bins=30, alpha=0.6, color="salmon",
         edgecolor="white", label=f"Merchant FOR Error (avg: {merchant_errors.mean():.2f} min)")
plt.hist(iot_errors, bins=30, alpha=0.6, color="green",
         edgecolor="white", label=f"IoT Sensor Error (avg: {iot_errors.mean():.2f} min)")
plt.xlabel("Absolute Error vs True KPT (minutes)")
plt.ylabel("Count")
plt.title("IoT Sensor vs Merchant FOR — Label Error Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/iot_error_distribution.png", dpi=150)
plt.close()
print("Chart saved: outputs/iot_error_distribution.png")

print("\nIoT Simulation complete.")
