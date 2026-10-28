from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from generate_dataset import generate_dataset

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "traffic_dataset.csv"
EXCEL_PATH = ROOT / "traffic_database.xlsx"
MODEL_PATH = ROOT / "traffic_model.pkl"
METRICS_PATH = ROOT / "model_metrics.json"
HISTORY_PATH = ROOT / "prediction_history.csv"

st.set_page_config(page_title="TrafficIQ | Smart Traffic", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: linear-gradient(145deg,#07111f 0%,#0b1728 50%,#10243a 100%); color:#e8f0fb;}
[data-testid="stSidebar"] {background:#081321;border-right:1px solid #1d3550;}
h1,h2,h3 {color:#f2f7ff;}
div[data-testid="stMetric"] {background:#10243a;border:1px solid #25435f;padding:16px;border-radius:14px;}
div.stButton > button {background:linear-gradient(90deg,#0ea5e9,#2563eb);color:white;border:0;border-radius:10px;padding:.55rem 1rem;font-weight:650;}
div[data-testid="stForm"] {background:#0e2034;padding:18px;border-radius:14px;border:1px solid #24415c;}
.small-muted {color:#9fb5ca;font-size:0.9rem;}
.hero {padding:22px 26px;border:1px solid #24415c;border-radius:18px;background:linear-gradient(120deg,#102b43,#0d1c2e);margin-bottom:18px;}
</style>
""", unsafe_allow_html=True)

def ensure_assets():
    if not DATA_PATH.exists() or not EXCEL_PATH.exists():
        generate_dataset()
    if not MODEL_PATH.exists():
        import train_model
        train_model.train()

ensure_assets()
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

def traffic_level(value):
    if value < 450: return "LOW", "🟢"
    if value < 850: return "MODERATE", "🟡"
    return "HIGH", "🔴"

def predict_one(road, hour, day_number, weather, holiday, temperature):
    is_weekend = int(day_number >= 5)
    sample = pd.DataFrame([{
        "Road": road, "Hour": hour, "Day_Number": day_number,
        "Is_Weekend": is_weekend, "Weather": weather,
        "Is_Holiday": int(holiday), "Temperature_C": temperature
    }])
    return max(0, int(round(float(model.predict(sample)[0]))))

# Lightweight demo login; no real authentication or sensitive data storage.
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = "Traffic Admin"

if not st.session_state.logged_in:
    st.markdown('<div class="hero"><h1>🚦 TrafficIQ</h1><p>Smart Traffic Volume Prediction & Relief Advisor</p><p class="small-muted">Machine-learning powered traffic insights for a smarter city.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Welcome to TrafficIQ")
        st.write("Predict traffic volume, understand rush-hour patterns and explore congestion-reduction suggestions.")
        st.info("Demo prototype using a generated synthetic dataset. It is not connected to live traffic sensors.")
    with right:
        with st.form("login_form"):
            st.subheader("Sign in to demo")
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", value="traffic123", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username.strip() and password:
                    st.session_state.logged_in = True
                    st.session_state.user_name = username.strip()
                    st.rerun()
                else:
                    st.error("Enter a username and password to continue.")
        st.caption("Demo login: admin / traffic123 (any non-empty credentials are accepted).")
    st.stop()

with st.sidebar:
    st.markdown("# 🚦 TrafficIQ")
    st.caption("SMART TRAFFIC INTELLIGENCE")
    st.write(f"Signed in as **{st.session_state.user_name}**")
    page = st.radio("NAVIGATION", ["Overview", "Predict Traffic", "Relief Advisor", "Data & Model"])
    st.divider()
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

st.markdown('<div class="hero"><h1>Smart Traffic Volume Prediction System</h1><p class="small-muted">Forecast • Understand • Recommend</p></div>', unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Traffic overview")
    total_rows = len(df)
    avg = int(df["Traffic_Volume"].mean())
    high_share = int((df["Traffic_Volume"] >= 850).mean() * 100)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset records", f"{total_rows:,}")
    c2.metric("Average vehicle count", f"{avg:,}")
    c3.metric("Roads simulated", df["Road"].nunique())
    c4.metric("High-volume observations", f"{high_share}%")
    hourly = df.groupby("Hour", as_index=False)["Traffic_Volume"].mean()
    fig = px.line(hourly, x="Hour", y="Traffic_Volume", markers=True, title="Average traffic volume by hour")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    road_avg = df.groupby("Road", as_index=False)["Traffic_Volume"].mean().sort_values("Traffic_Volume", ascending=False)
    fig2 = px.bar(road_avg, x="Road", y="Traffic_Volume", color="Traffic_Volume", title="Average traffic by road")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict Traffic":
    st.subheader("Predict traffic volume")
    st.write("Choose conditions to generate an ML-based estimate from the trained Random Forest model.")
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        roads = sorted(df["Road"].unique().tolist())
        road = a.selectbox("Road", roads)
        hour = b.slider("Hour of day", 0, 23, 18)
        day_name = c.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=4)
        d,e,f = st.columns(3)
        weather = d.selectbox("Weather", ["Clear","Cloudy","Rainy","Foggy"])
        holiday = e.selectbox("Holiday?", ["No","Yes"])
        temperature = f.slider("Temperature (°C)", 12, 39, 28)
        run = st.form_submit_button("Predict traffic")
    if run:
        day_number = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_name)
        volume = predict_one(road, hour, day_number, weather, holiday == "Yes", temperature)
        level, emoji = traffic_level(volume)
        st.markdown("---")
        m1,m2,m3 = st.columns(3)
        m1.metric("Predicted vehicles", f"{volume:,}")
        m2.metric("Congestion level", f"{emoji} {level}")
        m3.metric("Selected period", f"{day_name}, {hour:02d}:00")
        if level == "HIGH":
            st.error("Congestion alert: consider shifting departure time or checking another road's estimate.")
        elif level == "MODERATE":
            st.warning("Moderate traffic expected. Allow extra travel time if possible.")
        else:
            st.success("Lower predicted volume. Still check local conditions before travelling.")
        record = pd.DataFrame([{"Timestamp":pd.Timestamp.now().isoformat(timespec="seconds"),"Road":road,"Hour":hour,"Day":day_name,"Weather":weather,"Is_Holiday":holiday,"Temperature_C":temperature,"Predicted_Traffic":volume,"Traffic_Level":level}])
        if HISTORY_PATH.exists():
            old = pd.read_csv(HISTORY_PATH)
            record = pd.concat([old, record], ignore_index=True)
        record.to_csv(HISTORY_PATH, index=False)
        st.caption("This is a model estimate from synthetic training data, not a live traffic count.")

elif page == "Relief Advisor":
    st.subheader("Traffic Relief Advisor")
    st.write("Compare predicted volumes across simulated roads for the same time and conditions.")
    a,b,c = st.columns(3)
    hour = a.slider("Target hour", 0, 23, 18, key="relief_hour")
    day_name = b.selectbox("Target day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=4, key="relief_day")
    weather = c.selectbox("Weather", ["Clear","Cloudy","Rainy","Foggy"], key="relief_weather")
    holiday = st.selectbox("Holiday?", ["No","Yes"], key="relief_holiday")
    temp = st.slider("Temperature (°C)", 12, 39, 28, key="relief_temp")
    day_number = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_name)
    results = []
    for road in sorted(df["Road"].unique()):
        vol = predict_one(road, hour, day_number, weather, holiday == "Yes", temp)
        level, emoji = traffic_level(vol)
        results.append({"Road":road,"Predicted vehicles":vol,"Level":level})
    compare = pd.DataFrame(results).sort_values("Predicted vehicles")
    best = compare.iloc[0]
    st.dataframe(compare, use_container_width=True, hide_index=True)
    fig = px.bar(compare, x="Road", y="Predicted vehicles", color="Level", title="Predicted road comparison")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"Among the simulated roads, {best['Road']} has the lowest predicted volume ({int(best['Predicted vehicles']):,}) for these inputs.")
    st.caption("This is a model-based comparison, not a verified navigable route. Road distance, travel time and live conditions are not included.")

elif page == "Data & Model":
    st.subheader("Dataset explorer")
    st.write("Synthetic dataset generated for this PBL prototype.")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download CSV dataset", data=DATA_PATH.read_bytes(), file_name="traffic_dataset.csv", mime="text/csv")
    st.download_button("Download Excel database", data=EXCEL_PATH.read_bytes(), file_name="traffic_database.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.divider()
    st.subheader("Model evaluation")
    if metrics:
        a,b,c = st.columns(3)
        a.metric("MAE", metrics.get("MAE", "—"))
        b.metric("RMSE", metrics.get("RMSE", "—"))
        c.metric("R²", metrics.get("R2", "—"))
        st.caption(f"Test samples: {metrics.get('test_rows','—'):,} | Total dataset rows: {metrics.get('dataset_rows','—'):,}")
        st.info(metrics.get("note", ""))
    if HISTORY_PATH.exists():
        st.subheader("Recent prediction history")
        hist = pd.read_csv(HISTORY_PATH)
        st.dataframe(hist.tail(30).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("Prediction history will appear here after you run predictions.")
    st.download_button("Download prediction history (if available)", data=HISTORY_PATH.read_bytes() if HISTORY_PATH.exists() else b"", file_name="prediction_history.csv", mime="text/csv", disabled=not HISTORY_PATH.exists())
