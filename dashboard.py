import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#Color Palette
ZOMATO_RED = "#E23744"
DARK = "#1A1A1A"
LIGHT_GRAY = "#F8F8F8"
MID_GRAY = "#CCCCCC"
STEEL = "#4A90D9"

mpl.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

#Page Config
st.set_page_config(
    page_title="KPT Signal Enrichment — Team Big Bytes",
    page_icon="",
    layout="wide"
)

#Custom CSS
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #E23744; font-weight: 700; font-size: 1.9rem; }
    h2 { color: #1A1A1A; font-weight: 600; }
    h3 { color: #1A1A1A; font-weight: 600; font-size: 1.05rem; }

    div[data-testid="metric-container"] {
        background-color: #F8F8F8;
        border: 1px solid #EEEEEE;
        border-radius: 8px;
        padding: 1rem;
    }
    div[data-testid="metric-container"] label {
        color: #666666;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #E23744;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .section-divider {
        border: none;
        border-top: 2px solid #F0F0F0;
        margin: 1.5rem 0;
    }
    .tag {
        background-color: #FFF0F1;
        color: #E23744;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* --- Horizontal Nav Bar --- */
    .nav-bar {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 1.2rem 0 1.8rem 0;
        padding: 10px 14px;
        background: linear-gradient(90deg, #FFF0F1 0%, #FFE4E6 100%);
        border-radius: 12px;
        border: 1px solid #F5C6CA;
    }
    .nav-btn {
        background-color: white;
        color: #1A1A1A;
        border: 1.5px solid #E0E0E0;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        text-decoration: none;
        display: inline-block;
        user-select: none;
    }
    .nav-btn:hover {
        background-color: #FFF0F1;
        border-color: #E23744;
        color: #E23744;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(226,55,68,0.15);
    }
    .nav-btn.active {
        background-color: #E23744;
        border-color: #E23744;
        color: white;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(226,55,68,0.3);
    }

    /* Sidebar styling */
    .sidebar-section {
        background-color: #FFF0F1;
        border-left: 3px solid #E23744;
        padding: 10px 12px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
        font-size: 0.82rem;
        color: #1A1A1A;
        line-height: 1.5;
    }
    .sidebar-section-title {
        font-weight: 700;
        color: #E23744;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

#Load Data
@st.cache_data
def load_data():
    dirty = pd.read_csv("data/synthetic_orders.csv", parse_dates=["order_timestamp", "for_timestamp", "rider_arrival_timestamp"])
    clean = pd.read_csv("data/clean_orders.csv", parse_dates=["order_timestamp", "for_timestamp", "rider_arrival_timestamp"])
    enriched = pd.read_csv("data/enriched_orders.csv")
    external = pd.read_csv("data/external_kitchen_rush.csv")
    iot = pd.read_csv("data/iot_events.csv")
    return dirty, clean, enriched, external, iot

dirty_df, clean_df, enriched_df, external_df, iot_df = load_data()

#Session State for Navigation
if "page" not in st.session_state:
    st.session_state.page = "Overview"

#Sidebar
st.sidebar.markdown("""
<div style="background-color:#E23744; border-radius:10px; padding:10px 16px; margin-bottom:12px;">
    <span style="color:white; font-size:1.3rem; font-weight:800; letter-spacing:0.02em;">Zomathon 2026</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div class='sidebar-section'>
    <div class='sidebar-section-title'>The Problem</div>
    Merchant-marked FOR signals are systematically biased — merchants tap 'Food Ready'
    when the rider arrives, not when food is ready. This poisons KPT model training
    data across 300K+ merchants at scale.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class='sidebar-section'>
    <div class='sidebar-section-title'>Our Solution</div>
    A three-layer passive signal enrichment framework — GPS-based label debiasing,
    real-time Kitchen Load Score inference, and optional IoT hardware verification.
    Zero merchant friction. Deployable on Day 1.
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Team Big Bytes")
st.sidebar.markdown("**Arindam Pal** — Leader")
st.sidebar.markdown("Khushi Kumari")
st.sidebar.markdown("Sneha Tiwari")
st.sidebar.markdown("---")
st.sidebar.markdown("Birla Institute of Technology, Mesra")
st.sidebar.markdown("Lalpur Campus")
st.sidebar.markdown("---")

#Navigation Pages
PAGES = ["Overview", "Signal Validator", "Kitchen Load Score", "External Kitchen Rush", "KPT Model Results", "IoT Simulation"]

#Nav Button Logic
def nav_buttons():
    cols = st.columns(len(PAGES))
    for i, p in enumerate(PAGES):
        with cols[i]:
            if st.button(p, key=f"nav_{p}", use_container_width=True):
                st.session_state.page = p
                st.rerun()

#Header
st.title("Improving KPT Prediction at Zomato")
st.markdown("#### Signal Enrichment Framework — Three Layer Approach")
st.markdown('<span class="tag">Problem Statement 01</span>', unsafe_allow_html=True)

#Nav Bar
nav_buttons()

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

page = st.session_state.page


# PAGE 1: OVERVIEW

if page == "Overview":
    st.markdown("""
    Zomato's Kitchen Prep Time predictions are only as good as the data they're trained on.
    The current system relies on merchants manually tapping 'Food Ready' in the Mx app —
    but merchants often tap it when the rider arrives, not when food is actually ready.

    **Our solution fixes the data, not the model.**
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Orders Analyzed", f"{len(dirty_df):,}")
    col2.metric("Contaminated FOR Signals", f"{dirty_df['is_contaminated'].mean():.1%}")
    col3.metric("MAE Improvement", "33.2%")
    col4.metric("Peak Hour Underestimation", "13.99 min")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Layer 1 — Signal Cleaning")
        st.markdown("""
        GPS-based label debiasing. Detects when merchants tap
        'Food Ready' within 60 seconds of rider arrival and
        excludes these biased signals from training data.

        **Merchant effort: Zero**
        """)
    with col2:
        st.markdown("### Layer 2 — Kitchen Load Score")
        st.markdown("""
        Real-time kitchen stress inference from Mx app behavior —
        acceptance rate, response latency, rejection frequency,
        and concurrent orders combined into a 0-1 score.

        **Merchant effort: Zero**
        """)
    with col3:
        st.markdown("### Layer 3 — IoT Verification")
        st.markdown("""
        Low-cost ESP32 sensor at kitchen pass detects bag
        placement and sends a timestamped MQTT event —
        hardware-verified, zero-bias food-ready signal.

        **Merchant effort: One-time opt-in setup**
        """)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Key Results")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ["Baseline\n(Dirty Labels)", "Enriched\n(Clean + KLS)"]
        maes = [5.11, 3.42]
        colors = [MID_GRAY, ZOMATO_RED]
        bars = ax.bar(models, maes, color=colors, edgecolor="white", width=0.4)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mae:.2f} min", ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Mean Absolute Error (minutes)")
        ax.set_title("KPT Prediction Error: Before vs After Signal Enrichment")
        ax.set_ylim(0, 7)
        st.pyplot(fig)
        plt.close()

    with col2:
        hourly = external_df.groupby("hour")[["zomato_orders", "dine_in_orders", "competitor_orders"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly.index, hourly["zomato_orders"], label="Zomato Orders", color=ZOMATO_RED)
        ax.bar(hourly.index, hourly["dine_in_orders"], bottom=hourly["zomato_orders"],
               label="Dine-in Orders", color="#F4A261")
        ax.bar(hourly.index, hourly["competitor_orders"],
               bottom=hourly["zomato_orders"] + hourly["dine_in_orders"],
               label="Competitor Orders", color=MID_GRAY)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Orders")
        ax.set_title("Invisible Kitchen Load by Hour of Day")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()


# PAGE 2: SIGNAL VALIDATOR

elif page == "Signal Validator":
    st.markdown("## Signal Validator — GPS-Based Label Debiasing")
    st.markdown("""
    For every order, we compute the time delta between the merchant's FOR tap and
    the rider's GPS arrival at the restaurant. If the gap is under 60 seconds,
    the signal is classified as rider-influenced and excluded from model training.
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    dirty_df["for_rider_delta_seconds"] = (
        dirty_df["for_timestamp"] - dirty_df["rider_arrival_timestamp"]
    ).dt.total_seconds()
    dirty_df["is_rider_influenced"] = (dirty_df["for_rider_delta_seconds"].abs() <= 60).astype(int)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total FOR Signals", f"{len(dirty_df):,}")
    col2.metric("Rider-Influenced (Noisy)", f"{dirty_df['is_rider_influenced'].sum():,}",
                f"{dirty_df['is_rider_influenced'].mean():.1%} of total")
    col3.metric("Clean Signals Retained", f"{(dirty_df['is_rider_influenced']==0).sum():,}",
                f"{(dirty_df['is_rider_influenced']==0).mean():.1%} of total")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### FOR-Rider Time Delta Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(dirty_df["for_rider_delta_seconds"], bins=100, color=ZOMATO_RED,
                edgecolor="white", alpha=0.8)
        ax.axvline(x=60, color=DARK, linestyle="--", linewidth=2, label="60s threshold")
        ax.axvline(x=-60, color=DARK, linestyle="--", linewidth=2)
        ax.set_xlabel("FOR Timestamp minus Rider Arrival (seconds)")
        ax.set_ylabel("Number of Orders")
        ax.set_title("Spike near zero indicates rider-influenced signals")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Label Quality: Contaminated vs Clean")
        fig, ax = plt.subplots(figsize=(6, 4))
        contaminated = dirty_df[dirty_df["is_rider_influenced"] == 1]
        clean = dirty_df[dirty_df["is_rider_influenced"] == 0]
        ax.hist(contaminated["recorded_kpt_minutes"], bins=40, alpha=0.7,
                color=ZOMATO_RED, edgecolor="white", label="Contaminated Labels")
        ax.hist(clean["recorded_kpt_minutes"], bins=40, alpha=0.7,
                color=STEEL, edgecolor="white", label="Clean Labels")
        ax.axvline(dirty_df["true_kpt_minutes"].mean(), color=DARK,
                   linestyle="--", linewidth=2, label="True KPT Average")
        ax.set_xlabel("Recorded KPT (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("Contaminated labels skew training data significantly")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### Sample Rider-Influenced Signals Detected")
    sample = dirty_df[dirty_df["is_rider_influenced"] == 1][
        ["order_id", "restaurant_id", "for_rider_delta_seconds", "recorded_kpt_minutes", "true_kpt_minutes"]
    ].head(10).reset_index(drop=True)
    sample.columns = ["Order ID", "Restaurant", "FOR-Rider Delta (s)", "Recorded KPT (min)", "True KPT (min)"]
    st.dataframe(sample, use_container_width=True)


# PAGE 3: KITCHEN LOAD SCORE

elif page == "Kitchen Load Score":
    st.markdown("## Kitchen Load Score — Real-Time Kitchen Stress Inference")
    st.markdown("""
    The Kitchen Load Score (0 to 1) is computed per restaurant using order acceptance rate,
    Mx app response latency, rejection frequency, and concurrent active orders.
    A score near 1 indicates the kitchen is under heavy stress.
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Kitchen Load Score", f"{enriched_df['computed_kls'].mean():.3f}")
    col2.metric("Maximum Kitchen Load Score", f"{enriched_df['computed_kls'].max():.3f}")
    col3.metric("Correlation with True KPT", "0.561")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Kitchen Load Score vs True KPT")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(enriched_df["computed_kls"], enriched_df["true_kpt_minutes"],
                   alpha=0.3, color=ZOMATO_RED, s=10)
        ax.set_xlabel("Kitchen Load Score (0-1)")
        ax.set_ylabel("True KPT (minutes)")
        ax.set_title("Correlation 0.561 — KLS captures real kitchen stress")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Average Kitchen Load Score by Hour of Day")
        hourly_kls = enriched_df.groupby("order_hour")["computed_kls"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly_kls.index, hourly_kls.values, color=ZOMATO_RED,
               edgecolor="white", alpha=0.85)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Kitchen Load Score")
        ax.set_title("Peak hours show elevated kitchen load")
        st.pyplot(fig)
        plt.close()

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### Kitchen Load Score Distribution by Restaurant Size")
    fig, ax = plt.subplots(figsize=(9, 4))
    colors_map = {"small": STEEL, "medium": ZOMATO_RED, "large": "#2A9D8F"}
    for size in ["small", "medium", "large"]:
        subset = enriched_df[enriched_df["restaurant_size"] == size]["computed_kls"]
        ax.hist(subset, bins=30, alpha=0.65, label=size.capitalize(),
                color=colors_map[size], edgecolor="white")
    ax.set_xlabel("Kitchen Load Score")
    ax.set_ylabel("Count")
    ax.set_title("Larger restaurants carry higher kitchen load")
    ax.legend()
    st.pyplot(fig)
    plt.close()


# PAGE 4: EXTERNAL KITCHEN RUSH

elif page == "External Kitchen Rush":
    st.markdown("## External Kitchen Rush — The Invisible Load Problem")
    st.markdown("""
    Zomato's model only sees its own orders. But restaurants simultaneously serve dine-in
    customers and fulfill orders from competitor platforms. This invisible load causes
    systematic KPT underestimation — worst during peak hours.
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Zomato Orders / hr", f"{external_df['zomato_orders'].mean():.1f}")
    col2.metric("Avg Total Orders / hr", f"{external_df['total_orders'].mean():.1f}")
    col3.metric("Avg KPT Underestimation", f"{external_df['kpt_underestimation'].mean():.2f} min")
    col4.metric("Peak Hour Underestimation", "13.99 min")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Kitchen Load Breakdown by Hour")
        hourly = external_df.groupby("hour")[["zomato_orders", "dine_in_orders", "competitor_orders"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(hourly.index, hourly["zomato_orders"], label="Zomato Orders", color=ZOMATO_RED)
        ax.bar(hourly.index, hourly["dine_in_orders"], bottom=hourly["zomato_orders"],
               label="Dine-in Orders", color="#F4A261")
        ax.bar(hourly.index, hourly["competitor_orders"],
               bottom=hourly["zomato_orders"] + hourly["dine_in_orders"],
               label="Competitor Orders", color=MID_GRAY)
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Average Orders")
        ax.set_title("Zomato sees only the red portion")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### KPT Underestimation Gap by Hour")
        hourly_kpt = external_df.groupby("hour")[["true_kpt", "zomato_only_kpt"]].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hourly_kpt.index, hourly_kpt["true_kpt"], color=ZOMATO_RED,
                linewidth=2.5, marker="o", markersize=4, label="True KPT")
        ax.plot(hourly_kpt.index, hourly_kpt["zomato_only_kpt"], color=MID_GRAY,
                linewidth=2.5, marker="o", markersize=4, linestyle="--", label="Zomato-only estimate")
        ax.fill_between(hourly_kpt.index, hourly_kpt["zomato_only_kpt"],
                        hourly_kpt["true_kpt"], alpha=0.15, color=ZOMATO_RED,
                        label="Underestimation gap")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("KPT (minutes)")
        ax.set_title("Gap represents missed dine-in and competitor orders")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()


# PAGE 5: KPT MODEL RESULTS

elif page == "KPT Model Results":
    st.markdown("## KPT Model Results — Before vs After Signal Enrichment")
    st.markdown("""
    Two LightGBM models were trained — one on dirty biased labels (baseline),
    and one on GPS-cleaned labels with Kitchen Load Score as an additional feature.
    The improvement in prediction accuracy directly maps to Zomato's success metrics.
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline MAE", "5.11 min")
    col2.metric("Enriched MAE", "3.42 min")
    col3.metric("Error Reduction", "33.2%")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### MAE Comparison: Baseline vs Enriched")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Baseline\n(Dirty Labels)", "Enriched\n(Clean + KLS)"],
                      [5.11, 3.42], color=[MID_GRAY, ZOMATO_RED],
                      edgecolor="white", width=0.4)
        for bar, mae in zip(bars, [5.11, 3.42]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{mae:.2f} min", ha="center", fontweight="bold", fontsize=11)
        ax.set_ylabel("Mean Absolute Error (minutes)")
        ax.set_title("33.2% reduction in KPT prediction error")
        ax.set_ylim(0, 7)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Impact on Zomato Success Metrics")
        metrics = {
            "Metric": ["Rider Wait Time at Pickup", "ETA Prediction Error (P50)",
                       "Order Delay Rate", "Label Contamination Rate"],
            "Before": ["~8.5 min", "5.11 min", "High", "45%"],
            "After": ["~6.8 min", "3.42 min", "Reduced", "< 5%"],
            "Improvement": ["Down 1.7 min", "Down 33.2%", "Significant reduction", "Down 40pp"]
        }
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### Scalability — Deployment Scope")
    scale = {
        "Layer": ["Signal Cleaning (GPS)", "Kitchen Load Score", "IoT Sensors"],
        "Merchant Effort": ["None", "None", "Opt-in setup only"],
        "Deployment Scope": ["All 300K merchants — Day 1", "All 300K merchants", "Top 10K high-volume first"],
        "Infrastructure": ["Minor backend logic", "Kafka streaming pipeline", "MQTT broker + hardware rollout"]
    }
    st.dataframe(pd.DataFrame(scale), use_container_width=True)


# PAGE 6: IOT SIMULATION

elif page == "IoT Simulation":
    st.markdown("## IoT Simulation — ESP32 Kitchen Pass Sensor")
    st.markdown("""
    For high-volume partners, a low-cost ESP32 sensor at the kitchen pass detects
    when a food bag is placed on the counter. It sends a timestamped MQTT event —
    providing hardware-verified, zero-bias food-ready signals with no merchant action required.
    """)
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total IoT Events Captured", f"{len(iot_df):,}")
    col2.metric("IoT Label Error vs True KPT", "0.00 min")
    col3.metric("Merchant Label Error vs True KPT", "1.76 min")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Label Error Distribution: IoT vs Merchant FOR")
        fig, ax = plt.subplots(figsize=(6, 4))
        iot_errors = (iot_df["iot_kpt_minutes"] - iot_df["true_kpt_minutes"]).abs()
        merchant_errors = (iot_df["merchant_kpt_minutes"] - iot_df["true_kpt_minutes"]).abs()
        ax.hist(merchant_errors, bins=30, alpha=0.7, color=MID_GRAY,
                edgecolor="white", label=f"Merchant FOR (avg: {merchant_errors.mean():.2f} min)")
        ax.hist(iot_errors, bins=30, alpha=0.7, color=ZOMATO_RED,
                edgecolor="white", label=f"IoT Sensor (avg: {iot_errors.mean():.2f} min)")
        ax.set_xlabel("Absolute Error vs True KPT (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("IoT sensor provides near-perfect training labels")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Live Event Stream — Last 15 Events")
        display_iot = iot_df[["order_id", "iot_kpt_minutes", "merchant_kpt_minutes",
                               "true_kpt_minutes", "is_contaminated"]].tail(15).copy()
        display_iot["Signal Status"] = display_iot["is_contaminated"].map(
            {1: "Biased", 0: "Clean"})
        display_iot = display_iot.drop("is_contaminated", axis=1)
        display_iot.columns = ["Order ID", "IoT KPT", "Merchant KPT", "True KPT", "Signal Status"]
        st.dataframe(display_iot.reset_index(drop=True), use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("#### Hardware Specification")
    hw = {
        "Component": ["ESP32 Microcontroller", "Microphone Module", "Load Cell", "MQTT Broker"],
        "Estimated Cost": ["~Rs. 300", "~Rs. 50", "~Rs. 150", "Free (Mosquitto)"],
        "Purpose": [
            "Main controller with WiFi connectivity",
            "Acoustic detection of bag placement",
            "Weight-based food ready detection",
            "Lightweight IoT event messaging protocol"
        ]
    }
    st.dataframe(pd.DataFrame(hw), use_container_width=True)
