"""
AiCarer Health Monitor
Streamlit web application for health anomaly detection.
Target users: Supervisors/caregivers managing elderly residents in Brisbane.

Usage:
    pip install -r requirements.txt
    streamlit run aicarer_app.py
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import xgboost as xgb

# ── Constants ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = "AIzaSyD6Ut58D7WfLGTfuwTW1OGQ41ALlhfvNJ4"
LEAD_TIME      = 15  # minutes

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AiCarer Health Monitor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .aicarer-header {
        background: linear-gradient(135deg, #0d7377 0%, #14a085 50%, #1a7f8e 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 2rem; color: white;
    }
    .aicarer-header h1 { color: white; margin: 0; font-size: 2.2rem; }
    .aicarer-header p  { color: rgba(255,255,255,0.85); margin: 0.4rem 0 0; font-size: 1.05rem; }

    .risk-high   { background:#fee2e2; border-left:6px solid #ef4444; padding:24px 28px; border-radius:10px; margin:1rem 0; }
    .risk-medium { background:#fef3c7; border-left:6px solid #f59e0b; padding:24px 28px; border-radius:10px; margin:1rem 0; }
    .risk-low    { background:#d1fae5; border-left:6px solid #10b981; padding:24px 28px; border-radius:10px; margin:1rem 0; }
    .risk-high h2, .risk-medium h2, .risk-low h2 { margin:0 0 0.4rem; font-size:1.5rem; }
    .risk-high p,  .risk-medium p,  .risk-low p  { margin:0; font-size:1rem; }

    .result-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:18px 20px; height:100%; }
    .result-card h4 { color:#0d7377; margin:0 0 0.5rem; }
    .result-card p  { margin:0.2rem 0; color:#374151; font-size:0.95rem; }

    .upload-hint { background:#f0fdfa; border:1px solid #99f6e4; border-radius:8px; padding:14px 18px; margin-bottom:1rem; font-size:0.9rem; color:#134e4a; }
    hr { border:none; border-top:1px solid #e5e7eb; margin:2rem 0; }
    [data-testid="metric-container"] { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 18px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aicarer-header">
    <h1>❤️ AiCarer Health Monitor</h1>
    <p>Upload a patient's health data to run a full AI-powered analysis — risk alerts, trend charts, and caregiver guidance in one place.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — reset when user wants a new file
# ══════════════════════════════════════════════════════════════════════════════
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False
if "reset" not in st.session_state:
    st.session_state["reset"] = False

def reset_app():
    st.session_state["run_analysis"] = False
    st.session_state["reset"] = True

# ══════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE — runs once; won't re-run on UI interactions
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes):
    """Full 9-stage analysis pipeline. Cached so UI interactions don't retrigger it."""
    import io

    df = pd.read_csv(io.BytesIO(file_bytes))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ── Stage 1: Clean ────────────────────────────────────────────────────────
    df_clean = df.copy()
    df_clean["hr.mean"] = df_clean["hr.mean"].interpolate(method="linear", limit=5)
    df_clean = df_clean.dropna(subset=["hr.mean"])
    before_rr = len(df_clean)
    df_clean = df_clean[(df_clean["rr.mean"].isna()) | (df_clean["rr.mean"] <= 60)]
    removed_rr = before_rr - len(df_clean)
    data_days  = max((df_clean["timestamp"].max() - df_clean["timestamp"].min()).days, 1)
    last_reading = df_clean["timestamp"].max()

    # ── Stage 3: Z-score ──────────────────────────────────────────────────────
    window_size = 180
    df_clean["hr_rolling_mean"] = df_clean["hr.mean"].rolling(window_size, min_periods=1).mean()
    df_clean["hr_rolling_std"]  = df_clean["hr.mean"].rolling(window_size, min_periods=1).std()
    df_clean["z_score"] = (
        (df_clean["hr.mean"] - df_clean["hr_rolling_mean"])
        / (df_clean["hr_rolling_std"] + 0.0001)
    )
    anomalies = df_clean[abs(df_clean["z_score"]) > 3].copy()

    # ── Stage 6: Intelligent filtering ───────────────────────────────────────
    act_nonzero   = df_clean["act.mean"][df_clean["act.mean"] > 0]
    act_threshold = act_nonzero.quantile(0.25) if len(act_nonzero) > 0 else 0
    true_anomalies = anomalies[
        (anomalies["z_score"] > 3) & (anomalies["act.mean"] < act_threshold)
    ].copy()

    llm_payload = None
    if not true_anomalies.empty:
        latest = true_anomalies.iloc[-1]
        llm_payload = {
            "event_time":         latest["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "patient_status":     "Resting (Very Low Activity)",
            "current_hr":         int(latest["hr.mean"]),
            "normal_hr_baseline": int(latest["hr_rolling_mean"]),
            "z_score":            round(float(latest["z_score"]), 2),
            "stress_level":       "High" if latest["stress.mean"] > 75 else "Normal",
            "action_required":    True,
        }

    # ── Stage 8: Isolation Forest + One-Class SVM ─────────────────────────────
    features = ["hr.mean", "act.mean", "stress.mean", "hrv.mean", "rr.mean"]
    df_ml    = df_clean.dropna(subset=features).copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_ml[features])

    param_grid  = {"n_estimators": [50, 100, 200], "contamination": [0.01, 0.02, 0.05]}
    best_model, best_params, best_score = None, None, -float("inf")
    for params in ParameterGrid(param_grid):
        m = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            random_state=42,
        )
        m.fit(X_scaled)
        sv = m.decision_function(X_scaled).var()
        if sv > best_score:
            best_score, best_params, best_model = sv, params, m

    df_ml["if_anomaly"] = best_model.predict(X_scaled)
    if_count = (df_ml["if_anomaly"] == -1).sum()

    svm_model = OneClassSVM(kernel="rbf", gamma="auto", nu=best_params["contamination"])
    df_ml["svm_anomaly"] = svm_model.fit_predict(X_scaled)
    svm_count    = (df_ml["svm_anomaly"] == -1).sum()
    both_anomaly = ((df_ml["if_anomaly"] == -1) & (df_ml["svm_anomaly"] == -1)).sum()

    # ── Stage 9: XGBoost ──────────────────────────────────────────────────────
    df_ml["future_has_anomaly"] = (
        (df_ml["if_anomaly"] == -1).astype(int)
        .iloc[::-1].rolling(LEAD_TIME, min_periods=1).max().iloc[::-1]
    )
    df_ml["target"] = df_ml["future_has_anomaly"].astype(int)

    results = {}
    for window_label, WIN in [("30min", 30), ("60min", 60)]:
        pred_feats = []
        for col in features:
            df_ml[f"{col}_rmean_{WIN}"] = df_ml[col].rolling(WIN, min_periods=1).mean()
            df_ml[f"{col}_rstd_{WIN}"]  = df_ml[col].rolling(WIN, min_periods=1).std()
            pred_feats.extend([f"{col}_rmean_{WIN}", f"{col}_rstd_{WIN}"])

        df_p = df_ml.dropna(subset=pred_feats + ["target"]).copy()
        X, y = df_p[pred_feats].values, df_p["target"].values
        tscv = TimeSeriesSplit(n_splits=5)
        reports = []
        for _, (tr, te) in enumerate(tscv.split(X), 1):
            Xt, Xe, yt, ye = X[tr], X[te], y[tr], y[te]
            neg, pos = (yt == 0).sum(), (yt == 1).sum()
            clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                scale_pos_weight=neg / (pos + 1e-6),
                random_state=42, eval_metric="logloss",
            )
            clf.fit(Xt, yt)
            reports.append(
                classification_report(ye, clf.predict(Xe), output_dict=True, zero_division=0)
            )
        avg_f1 = float(np.mean([r["macro avg"]["f1-score"] for r in reports]))
        results[window_label] = {"avg_macro_f1": avg_f1, "model": clf, "features": pred_feats}

    best_window  = max(results, key=lambda k: results[k]["avg_macro_f1"])
    final_model  = results[best_window]["model"]
    df_pred_fin  = df_ml.dropna(subset=results[best_window]["features"] + ["target"]).copy()
    X_fin        = df_pred_fin[results[best_window]["features"]].values
    current_risk = float(final_model.predict_proba(X_fin[-1:, :])[0][1])

    return {
        "df_clean":    df_clean,
        "anomalies":   anomalies,
        "removed_rr":  removed_rr,
        "data_days":   data_days,
        "last_reading": last_reading,
        "llm_payload": llm_payload,
        "if_count":    int(if_count),
        "svm_count":   int(svm_count),
        "both_anomaly": int(both_anomaly),
        "df_ml_len":   len(df_ml),
        "results":     {k: {"avg_macro_f1": v["avg_macro_f1"]} for k, v in results.items()},
        "best_window": best_window,
        "current_risk": current_risk,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD + PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

# If showing results, offer a reset button instead of the uploader
if st.session_state["run_analysis"]:
    st.button("🔄 Analyse a New File", on_click=reset_app, type="secondary")
    st.markdown("---")
else:
    # ── Patient name ──────────────────────────────────────────────────────────
    patient_name = st.text_input(
        "Patient name or ID",
        placeholder="e.g. Wayne Smith",
        help="This label will appear on all results so you know whose data you are viewing.",
    )
    st.session_state["patient_name"] = patient_name

    # ── Upload hint ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="upload-hint">
        📋 <strong>Required CSV columns:</strong>
        <code>timestamp</code> &nbsp;·&nbsp;
        <code>hr.mean</code> (heart rate) &nbsp;·&nbsp;
        <code>rr.mean</code> (respiratory rate) &nbsp;·&nbsp;
        <code>stress.mean</code> &nbsp;·&nbsp;
        <code>hrv.mean</code> (heart rate variability) &nbsp;·&nbsp;
        <code>act.mean</code> (activity level)
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop the patient's health CSV file here, or click to browse",
        type=["csv"],
    )
    st.session_state["uploaded_file"] = uploaded_file

    if uploaded_file is not None:
        # ── Data preview ──────────────────────────────────────────────────────
        try:
            preview_bytes = uploaded_file.read()
            uploaded_file.seek(0)
            preview_df = pd.read_csv(pd.io.common.BytesIO(preview_bytes))

            required_cols = {"timestamp", "hr.mean", "rr.mean", "stress.mean", "hrv.mean", "act.mean"}
            missing = required_cols - set(preview_df.columns)

            st.markdown(f"**📄 File:** `{uploaded_file.name}` &nbsp;·&nbsp; **{len(preview_df):,} rows** &nbsp;·&nbsp; {len(preview_df.columns)} columns")

            if missing:
                st.error(f"⛔ Missing required columns: {', '.join(sorted(missing))}. Please check the file and re-upload.")
            else:
                st.success("✅ File looks good — all required columns detected.")
                st.caption("Preview of first 3 rows:")
                st.dataframe(preview_df.head(3), use_container_width=True)

                st.session_state["file_bytes"]  = preview_bytes
                st.session_state["file_valid"]  = True

        except Exception as e:
            st.error(f"Could not read this file: {e}")
            st.session_state["file_valid"] = False
    else:
        st.info("👆 Upload a CSV file above to get started.")
        st.stop()

    if not st.session_state.get("file_valid"):
        st.stop()

    if st.button("▶ Run Full Analysis", type="primary", use_container_width=True):
        st.session_state["run_analysis"] = True
        st.rerun()

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

patient_name = st.session_state.get("patient_name", "")
file_bytes   = st.session_state.get("file_bytes")

if not file_bytes:
    st.warning("Session expired — please upload the file again.")
    reset_app()
    st.rerun()

# ── Run (cached) pipeline ─────────────────────────────────────────────────────
with st.spinner("Running analysis pipeline — this may take up to 60 seconds on first run…"):
    R = run_pipeline(file_bytes)

progress = st.progress(100, text="Analysis complete.")

df_clean     = R["df_clean"]
anomalies    = R["anomalies"]
last_reading = R["last_reading"]

# ── Patient banner ────────────────────────────────────────────────────────────
if patient_name:
    st.markdown(f"### 👤 Patient: {patient_name}")

# ── Data Summary ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Data Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Readings", f"{len(df_clean):,}")
c2.metric("Days of Data",   f"{R['data_days']}")
c3.metric("Last Reading",   last_reading.strftime("%d %b %Y %H:%M"))
c4.metric("Sensor Errors Removed", f"{R['removed_rr']:,}")
st.success(f"✅ {len(df_clean):,} clean records loaded — covering {R['data_days']} days.")

# ── 7-day HR trend ─────────────────────────────────────────────────────────────
last_7   = last_reading - pd.Timedelta(days=7)
df_7     = df_clean[df_clean["timestamp"] >= last_7]
anom_7   = anomalies[anomalies["timestamp"] >= last_7]

st.markdown("---")
st.subheader("📈 Heart Rate — Last 7 Days")

fig_hr = go.Figure()
fig_hr.add_trace(go.Scatter(
    x=df_7["timestamp"], y=df_7["hr.mean"],
    mode="lines", name="Heart Rate (bpm)",
    line=dict(color="#0d7377", width=1.5),
    hovertemplate="%{x|%d %b %H:%M}<br>Heart Rate: %{y} bpm<extra></extra>",
))
fig_hr.add_trace(go.Scatter(
    x=df_7["timestamp"],
    y=df_7["hr.mean"].rolling(30, min_periods=1).mean(),
    mode="lines", name="30-min average",
    line=dict(color="#14a085", width=2, dash="dot"),
    hovertemplate="%{x|%d %b %H:%M}<br>30-min avg: %{y:.1f} bpm<extra></extra>",
))
fig_hr.update_layout(
    xaxis_title="Date & Time", yaxis_title="Heart Rate (bpm)",
    hovermode="x unified", height=360,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.08),
    plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
)
st.plotly_chart(fig_hr, use_container_width=True)

# ── Anomaly chart ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Unusual Heart Rate Events — Last 7 Days")
st.caption(
    f"The orange line is the patient's personal 3-hour rolling average. "
    f"**Red stars** mark moments when the heart rate moved significantly outside that personal baseline. "
    f"**{len(anom_7):,}** such events found in the last 7 days."
)
fig_anom = go.Figure()
fig_anom.add_trace(go.Scatter(
    x=df_7["timestamp"], y=df_7["hr.mean"],
    mode="lines", name="Heart Rate",
    line=dict(color="#93c5fd", width=1.5),
    hovertemplate="%{x|%d %b %H:%M}<br>HR: %{y} bpm<extra></extra>",
))
fig_anom.add_trace(go.Scatter(
    x=df_7["timestamp"], y=df_7["hr_rolling_mean"],
    mode="lines", name="Personal baseline (3-hr avg)",
    line=dict(color="#f97316", width=2),
    hovertemplate="%{x|%d %b %H:%M}<br>Baseline: %{y:.1f} bpm<extra></extra>",
))
if not anom_7.empty:
    fig_anom.add_trace(go.Scatter(
        x=anom_7["timestamp"], y=anom_7["hr.mean"],
        mode="markers", name="Unusual event",
        marker=dict(color="#ef4444", size=11, symbol="star"),
        hovertemplate="%{x|%d %b %H:%M}<br>HR: %{y} bpm<extra></extra>",
    ))
fig_anom.update_layout(
    xaxis_title="Date & Time", yaxis_title="Heart Rate (bpm)",
    hovermode="x unified", height=360,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", y=1.08),
    plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
)
st.plotly_chart(fig_anom, use_container_width=True)

# ── 14-day daily anomaly bar chart ────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Daily Unusual Events — Last 14 Days")
st.caption(
    "Day-by-day counts give you a clearer picture than weekly totals — "
    "you can see exactly which day was busy rather than which week."
)

last_14   = last_reading - pd.Timedelta(days=14)
anom_14   = anomalies[anomalies["timestamp"] >= last_14].copy()
anom_14["date"] = anom_14["timestamp"].dt.date

date_range   = pd.date_range(last_14.date(), last_reading.date())
daily_counts = (
    anom_14.groupby("date").size()
    .reindex(date_range.date, fill_value=0)
    .reset_index()
)
daily_counts.columns = ["date", "count"]

# Fixed colour thresholds: 0 → green, 1-3 → amber, 4+ → red
def bar_color(n):
    if n == 0:   return "#10b981"
    if n <= 3:   return "#f59e0b"
    return "#ef4444"

daily_counts["color"] = daily_counts["count"].apply(bar_color)

fig_14 = go.Figure(go.Bar(
    x=daily_counts["date"].astype(str),
    y=daily_counts["count"],
    marker_color=daily_counts["color"],
    hovertemplate="%{x}<br>Unusual events: %{y}<extra></extra>",
))
fig_14.update_layout(
    xaxis_title="Date", yaxis_title="Unusual Events",
    height=260, margin=dict(l=0, r=0, t=10, b=0),
    plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
)
# Colour legend annotation
fig_14.add_annotation(
    text="🟢 0   🟡 1–3   🔴 4+",
    xref="paper", yref="paper", x=1, y=1.12,
    showarrow=False, font=dict(size=11), align="right",
)
st.plotly_chart(fig_14, use_container_width=True)

# ── Analysis Results ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Analysis Results")

if_count    = R["if_count"]
both_anomaly = R["both_anomaly"]
svm_count   = R["svm_count"]
total_ml    = R["df_ml_len"]
results_sum = R["results"]
best_window = R["best_window"]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="result-card">
        <h4>🔬 Multi-sensor check</h4>
        <p>Found <strong>{if_count:,}</strong> unusual patterns across all sensors
        out of <strong>{total_ml:,}</strong> total readings
        (<strong>{if_count/total_ml*100:.1f}%</strong>).</p>
        <p style="color:#6b7280;font-size:0.85rem;margin-top:8px;">
        Analyses 5 sensors simultaneously: heart rate, activity, stress, HRV, and breathing rate.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    overlap_pct = (both_anomaly / if_count * 100) if if_count > 0 else 0
    st.markdown(f"""
    <div class="result-card">
        <h4>✅ Confirmed by second check</h4>
        <p>Both AI systems independently agreed on
        <strong>{both_anomaly:,}</strong> cases
        ({overlap_pct:.0f}% overlap) — these are the <strong>highest-confidence alerts</strong>.</p>
        <p style="color:#6b7280;font-size:0.85rem;margin-top:8px;">
        A second independent model cross-checks which patterns are most significant.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    best_f1  = results_sum[best_window]["avg_macro_f1"]
    lookback = "past 60 minutes" if best_window == "60min" else "past 30 minutes"
    if best_f1 >= 0.7:
        quality       = "Good"
        quality_color = "#10b981"
        quality_note  = "The AI performed well on historical data."
    elif best_f1 >= 0.6:
        quality       = "Acceptable"
        quality_color = "#f59e0b"
        quality_note  = "The AI gave useful results, though not perfect."
    else:
        quality       = "Limited"
        quality_color = "#ef4444"
        quality_note  = "Treat this prediction with caution."

    st.markdown(f"""
    <div class="result-card">
        <h4>🎯 How reliable is the prediction?</h4>
        <p>The AI reviews the <strong>{lookback}</strong> of readings to forecast
        the next 15 minutes.</p>
        <p>Prediction quality:
            <strong style="color:{quality_color};">{quality}</strong>
        </p>
        <p style="color:#6b7280;font-size:0.85rem;margin-top:8px;">
        {quality_note}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Risk Alert ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Risk Level at Last Reading")

current_risk = R["current_risk"]
reading_time = last_reading.strftime("%d %b %Y at %H:%M")
pct          = f"{current_risk:.1%}"

if current_risk > 0.70:
    st.markdown(f"""
    <div class="risk-high">
        <h2>⚠️  HIGH ALERT</h2>
        <p><strong>{pct} probability of an unusual health event in the 15 minutes following the last recorded reading
        ({reading_time}).</strong></p>
        <p>Please check in with the patient and follow your organisation's monitoring protocol.</p>
    </div>
    """, unsafe_allow_html=True)
    alert_level = "HIGH ALERT"

elif current_risk > 0.40:
    st.markdown(f"""
    <div class="risk-medium">
        <h2>⚡  WATCH CLOSELY</h2>
        <p><strong>{pct} probability of an unusual health event in the 15 minutes following the last recorded reading
        ({reading_time}).</strong></p>
        <p>No immediate action required. Continue routine monitoring as per your organisation's guidelines.</p>
    </div>
    """, unsafe_allow_html=True)
    alert_level = "WATCH CLOSELY"

else:
    st.markdown(f"""
    <div class="risk-low">
        <h2>✅  ALL LOOKS STABLE</h2>
        <p><strong>{pct} probability of an unusual health event in the 15 minutes following the last recorded reading
        ({reading_time}).</strong></p>
        <p>Readings are within the patient's normal range. Continue routine monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    alert_level = "STABLE"

# ── Copy Alert Summary (item 10) ──────────────────────────────────────────────
patient_label = f" — {patient_name}" if patient_name else ""
alert_summary = (
    f"AiCarer Health Monitor Report{patient_label}\n"
    f"{'='*45}\n"
    f"Generated from data recorded up to: {reading_time}\n"
    f"Risk status: {alert_level}\n"
    f"Risk probability (next 15 min): {pct}\n"
    f"Unusual events detected (last 7 days): {len(anom_7):,}\n"
    f"High-confidence alerts (both AI systems agreed): {both_anomaly:,}\n"
    f"{'='*45}\n"
    f"This report was generated by AiCarer Health Monitor.\n"
    f"Please follow your organisation's monitoring protocol for next steps."
)

with st.expander("📋 Copy Alert Summary", expanded=(current_risk > 0.40)):
    st.code(alert_summary, language=None)
    st.caption("Select all text above and copy it to share with your team or include in a handover note.")

# ── Gemini AI Caregiver Summary ────────────────────────────────────────────────
st.markdown("---")
st.subheader("💬 AI Caregiver Guidance")

llm_payload = R["llm_payload"]

if llm_payload is None:
    st.success("✅ No resting-state alert detected. No AI caregiver message is needed at this time.")
elif not GOOGLE_API_KEY:
    st.warning("AI guidance is unavailable — no API key configured.")
else:
    with st.spinner("Generating caregiver guidance from AI…"):
        try:
            from google import genai  # type: ignore

            client      = genai.Client(api_key=GOOGLE_API_KEY)
            models_list = [m.name for m in client.models.list()]
            target_model = next(
                (m for m in models_list if "gemini-2.5-flash" in m or "gemini-1.5-flash" in m),
                models_list[0] if models_list else None,
            )

            if target_model:
                system_prompt = (
                    "You are a helpful assistant for the AiCarer platform, supporting non-technical caregivers. "
                    "Your role is to summarise wearable sensor data in plain, reassuring language. "
                    "You do NOT provide medical advice, diagnoses, or treatment recommendations of any kind. "
                    "Always direct caregivers to follow their organisation's internal protocols and, "
                    "where appropriate, to contact the relevant healthcare professional directly.\n\n"
                    "Format your response as:\n"
                    "1. One plain-English sentence describing what the data shows.\n"
                    "2. Brief context: what is normal for this patient vs. what was observed.\n"
                    "3. One suggested non-clinical action for the caregiver (e.g. check in, log the event, notify a supervisor).\n"
                    "Use British English. Keep the tone calm and factual."
                )
                response = client.models.generate_content(
                    model=target_model,
                    contents=(
                        f"{system_prompt}\n\n"
                        f"Latest sensor event summary:\n{json.dumps(llm_payload, indent=2)}"
                    ),
                )
                st.info(f"🤖 **AiCarer AI Summary**\n\n{response.text}")
            else:
                st.warning("AI model unavailable — skipping guidance.")

        except Exception as e:
            st.warning(
                f"AI guidance could not be generated (API error: {e}). "
                "Please check your internet connection and try again."
            )

st.balloons()
st.success("🎉 Analysis complete! Scroll up to review all results.")
