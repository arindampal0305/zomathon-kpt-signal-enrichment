# Zomathon 2026 — KPT Signal Enrichment Framework
### Improving Kitchen Prep Time Prediction at Zomato

**Team Big Bytes**
Birla Institute of Technology, Mesra (Lalpur Campus)

| Member | Role |
|---|---|
| Arindam Pal | Team Leader |
| Khushi Kumari | Member |
| Sneha Tiwari | Member |

---

## Problem Statement

Zomato's Kitchen Prep Time (KPT) prediction relies on merchant-marked Food Order Ready (FOR) signals from the Mx app. These signals are systematically biased: merchants often tap 'Food Ready' when the rider arrives, not when food is actually prepared. This contaminates model training data at scale across 300,000+ merchant partners.

**Core issues:**
- 45% of FOR signals are rider-influenced (tapped within 60 seconds of rider arrival)
- Kitchen load from dine-in and competitor platform orders is completely invisible to Zomato
- No amount of model tuning can fix predictions trained on biased ground truth labels

---

## Our Solution: Three Layer Signal Enrichment Framework

> "We don't make the model smarter, We make the data honest."

Rather than improving the prediction model, we fix the input signals feeding it. Our solution introduces three passive layers of signal enrichment with zero additional friction for merchants.

### Layer 1: GPS-Based Label Debiasing
For every order, we compute the time delta between the merchant's FOR tap and the rider's GPS arrival at the restaurant. If the gap is under 60 seconds, the signal is flagged as rider-influenced and excluded from model training.

- Deployment scope: All 300K merchants, Day 1
- Merchant effort: Zero
- Result: 58.6% of noisy labels identified and removed

### Layer 2: Kitchen Load Score
A real-time Kitchen Load Score (0-1) is computed per restaurant using Mx app behavioral signals: order acceptance rate, response latency, rejection frequency, and concurrent active order count. This score captures kitchen stress from dine-in and competitor orders without any merchant action.

- Deployment scope: All 300K merchants
- Merchant effort: Zero
- Result: 0.561 correlation with true KPT: validates real kitchen stress capture

### Layer 3: IoT Hardware Verification (Optional, but recommended)
A low-cost ESP32 microcontroller (~Rs. 500 total) at the kitchen pass detects bag placement via acoustic or weight sensing and sends a timestamped MQTT event, providing hardware-verified, zero-bias food-ready signals for high-volume partners.

- Deployment scope: Top 10K high-volume merchants first
- Merchant effort: One-time opt-in setup
- Result: 0.00 min label error vs true KPT (vs 1.76 min for merchant FOR)

---

## Key Results

| Metric | Before | After | Improvement |
|---|---|---|---|
| KPT MAE | 5.11 min | 3.42 min | 33.2% reduction |
| Rider Wait Time | ~8.5 min | ~6.8 min | Down 1.7 min |
| Label Contamination Rate | 45% | < 5% | Down 40pp |
| Peak Hour Underestimation | 13.99 min | Captured via KLS | Addressed |

---

## External Kitchen Rush Finding

Zomato currently sees only 5.5 orders/hour per restaurant on average, while the kitchen is actually handling 13.1 orders/hour total: meaning **58% of kitchen load is completely invisible** to the current system. During peak hours, this causes KPT underestimation of up to 13.99 minutes.

---

## Project Structure
```
zomathon/
├── src/                        # All Python scripts
│   ├── generate_dataset.py     # Synthetic dataset generation
│   ├── signal_validator.py     # Layer 1 — GPS label debiasing
│   ├── kitchen_load_score.py   # Layer 2 — Kitchen Load Score
│   ├── external_kitchen_rush.py # External load simulation
│   ├── kpt_model.py            # KPT model training and comparison
│   └── iot_simulator.py        # Layer 3 — IoT sensor simulation
├── data/                       # Generated datasets (CSV)
├── outputs/                    # Generated charts (PNG)
├── dashboard.py                # Streamlit interactive dashboard
├── .streamlit/config.toml      # Dashboard theme configuration
└── README.md                   # This file
```

---

## Setup and Installation

**Requirements:** Python 3.10+

**Step 1> Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/zomathon-kpt-signal-enrichment.git
cd zomathon-kpt-signal-enrichment
```

**Step 2> Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**Step 3> Install dependencies**
```bash
pip install pandas numpy lightgbm scikit-learn streamlit matplotlib seaborn faker
```

**Step 4> Run all scripts in order**
```bash
python src/generate_dataset.py
python src/signal_validator.py
python src/kitchen_load_score.py
python src/external_kitchen_rush.py
python src/kpt_model.py
python src/iot_simulator.py
```

**Step 5> Launch the dashboard**
```bash
streamlit run dashboard.py
```

---

## Technology Stack:::

| Layer | Tool | Purpose |
|---|---|---|
| Backend | Python + FastAPI | Core logic and API |
| Stream Processing | Apache Kafka | Real-time Mx app event processing |
| ML Model | LightGBM | KPT prediction |
| Feature Cache | Redis | Sub-ms Kitchen Load Score serving |
| Database | PostgreSQL | Label store and signal quality logs |
| IoT Protocol | MQTT + Mosquitto | Lightweight IoT event messaging |
| IoT Hardware | ESP32 + sensors | Kitchen pass bag detection |
| Dashboard | Streamlit | Interactive ops dashboard |

---

## Scalability

| Layer | Merchant Effort | Scope | Timeline |
|---|---|---|---|
| Signal Cleaning | None | All 300K merchants | Day 1 |
| Kitchen Load Score | None | All 300K merchants | 1 week engineering |
| IoT Sensors | Opt-in setup | Top 10K high-volume first | Phased rollout |

---

## Success Metrics Addressed

- Average rider wait time at pickup
- ETA prediction error (P50 / P90)
- Order delay and cancellation rates
- Rider idle time

---

*Submission for Zomathon 2026 — Problem Statement 01*
*Team Big Bytes | BIT Mesra, Lalpur Campus*