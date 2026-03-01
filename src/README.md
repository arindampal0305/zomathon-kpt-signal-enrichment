# src : Source Scripts

This folder contains all Python scripts for the KPT Signal Enrichment Framework.
Each script corresponds to a specific layer or component of the solution and should
be run in the order listed below.

---

## Execution Order
```bash
python src/generate_dataset.py
python src/signal_validator.py
python src/kitchen_load_score.py
python src/external_kitchen_rush.py
python src/kpt_model.py
python src/iot_simulator.py
```

---

## Script Reference

### generate_dataset.py
Generates a synthetic dataset of 5,000 food delivery orders across 50 restaurants.
Simulates realistic order timestamps, FOR tap timestamps, rider GPS arrival times,
concurrent order counts, Mx app response latency, and rejection flags.

Key output : `data/synthetic_orders.csv`

Contamination logic : 45% of FOR signals are deliberately seeded as rider-influenced
(tapped within 60 seconds of rider arrival) to simulate the real-world bias problem.

---

### signal_validator.py
Implements Layer 1 of the signal enrichment framework : GPS-based label debiasing.

For each order, computes the time delta between the merchant's FOR tap timestamp
and the rider's GPS arrival timestamp at the restaurant. Any FOR signal tapped
within 60 seconds of rider arrival is classified as rider-influenced and excluded
from the training label set.

Key output : `data/clean_orders.csv`

Key result : 58.6% of FOR signals identified as rider-influenced and removed.
Bias introduced by contamination : 0.93 minutes on average.

---

### kitchen_load_score.py
Implements Layer 2 of the signal enrichment framework : Kitchen Load Score computation.

Computes a real-time Kitchen Load Score (0 to 1) per restaurant using a rolling
window of Mx app behavioral signals : order acceptance rate, response latency,
rejection frequency, and concurrent active order count. Normalized and weighted
into a single composite score.

Key output : `data/enriched_orders.csv`

Key result : 0.561 correlation between Kitchen Load Score and true KPT,
validating that the score captures genuine kitchen stress.

---

### external_kitchen_rush.py
Simulates the invisible kitchen load problem : orders from dine-in customers and
competitor platforms (Swiggy, phone orders) that consume kitchen capacity but are
completely invisible to Zomato's current system.

Key output : `data/external_kitchen_rush.csv`

Key result : Zomato sees only 5.5 orders/hour on average while the kitchen handles
13.1 orders/hour total. Peak hour KPT underestimation reaches 13.99 minutes.

---

### kpt_model.py
Trains two LightGBM models and compares their performance :

- Baseline model : trained on dirty, contaminated labels with no Kitchen Load Score
- Enriched model : trained on GPS-cleaned labels with Kitchen Load Score as a feature

Evaluates both models against true KPT values on a held-out test set.

Key result : 33.2% reduction in Mean Absolute Error (5.11 min down to 3.42 min).

---

### iot_simulator.py
Implements Layer 3 of the signal enrichment framework : IoT sensor simulation.

Simulates an ESP32 microcontroller at the kitchen pass that detects bag placement
and sends timestamped MQTT events via a Mosquitto broker. Demonstrates the label
quality improvement from hardware-verified food-ready signals compared to
merchant-tapped FOR signals.

Key output : `data/iot_events.csv`

Key result : IoT label error of 0.00 minutes vs true KPT, compared to 1.76 minutes
for merchant FOR signals.

---

*Team Big Bytes | Zomathon 2026 | BIT Mesra, Lalpur Campus*