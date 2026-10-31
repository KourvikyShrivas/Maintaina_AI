"""Maintaina AI — predictive maintenance dashboard with light/dark themes."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests
import streamlit as st

from maintaina.config import KNOWLEDGE_DIR, MODEL_DIR
from maintaina.predict import DISCLAIMER, PredictionService
from maintaina.rag import KnowledgeBase, build_query, guidance
from maintaina.schemas import SensorReading

API_URL = os.getenv("API_URL", "").rstrip("/")

st.set_page_config(
    page_title="Maintaina AI",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Theme ----------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

with st.sidebar:
    st.markdown("## 🛠️ Maintaina AI")
    st.caption("Predictive maintenance workspace")
    theme_choice = st.radio(
        "Appearance",
        ["Light", "Dark"],
        index=0 if st.session_state.theme == "Light" else 1,
        horizontal=True,
        key="theme_picker",
    )
    st.session_state.theme = theme_choice

DARK = st.session_state.theme == "Dark"
if DARK:
    BG = "#0B1220"
    PANEL = "#111C2E"
    PANEL_ALT = "#162338"
    TEXT = "#E8EEF7"
    MUTED = "#9AAAC0"
    BORDER = "#26364D"
    ACCENT = "#31C6B4"
    ACCENT_SOFT = "#123D42"
    INPUT = "#17263A"
else:
    BG = "#F4F7FB"
    PANEL = "#FFFFFF"
    PANEL_ALT = "#F8FAFD"
    TEXT = "#152238"
    MUTED = "#64748B"
    BORDER = "#E3EAF2"
    ACCENT = "#0F9D91"
    ACCENT_SOFT = "#E2F5F2"
    INPUT = "#FFFFFF"

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {"dark" if DARK else "light"}; }}
    .stApp {{ background: {BG}; color: {TEXT}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] * {{ color: {TEXT}; }}
    .block-container {{ padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1500px; }}
    h1, h2, h3, h4, p, label, .stMarkdown {{ color: {TEXT}; }}
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid {BORDER}; padding: 17px 18px;
        border-radius: 15px; box-shadow: 0 3px 14px rgba(15, 35, 60, 0.035);
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ color: {TEXT}; }}
    div[data-testid="stForm"], div[data-testid="stExpander"], div[data-testid="stDataFrame"] {{
        border-color: {BORDER};
    }}
    div[data-testid="stForm"] {{
        background: {PANEL}; border: 1px solid {BORDER}; padding: 18px; border-radius: 16px;
    }}
    div[data-testid="stExpander"] {{ background: {PANEL}; border-radius: 12px; }}
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
        background: {INPUT}; color: {TEXT}; border-color: {BORDER};
    }}
    .stTextArea textarea {{
    background-color: {INPUT} !important;
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
    caret-color: {TEXT} !important;
}}

.stTextArea textarea::placeholder {{
    color: {MUTED} !important;
    -webkit-text-fill-color: {MUTED} !important;
    opacity: 1;
}}
    .stButton > button, .stFormSubmitButton > button {{
        border-radius: 10px; border: 1px solid {ACCENT}; font-weight: 650;
    }}
    .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
        background: {ACCENT}; color: white;
    }}
    .hero {{
        background: linear-gradient(120deg, {PANEL}, {PANEL_ALT});
        border: 1px solid {BORDER}; border-radius: 20px; padding: 25px 28px;
        margin-bottom: 20px;
    }}
    .eyebrow {{ color: {ACCENT}; font-size: 0.76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
    .hero-title {{ color: {TEXT}; font-size: 2rem; font-weight: 800; margin: 5px 0 4px 0; }}
    .hero-sub {{ color: {MUTED}; font-size: .98rem; margin: 0; }}
    .section-title {{ color: {TEXT}; font-size: 1.12rem; font-weight: 750; margin: 8px 0 12px 0; }}
    .info-card {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 15px;
        padding: 17px 18px; min-height: 118px;
    }}
    .info-label {{ color: {MUTED}; font-size: .82rem; margin-bottom: 8px; }}
    .info-value {{ color: {TEXT}; font-size: 1.55rem; font-weight: 800; }}
    .info-note {{ color: {MUTED}; font-size: .78rem; margin-top: 5px; }}
    .pill {{
        display: inline-block; background: {ACCENT_SOFT}; color: {ACCENT};
        border-radius: 99px; padding: 5px 10px; font-size: .76rem; font-weight: 750;
    }}
    .muted {{ color: {MUTED}; }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Existing model and knowledge services ----------
@st.cache_resource(show_spinner="Loading the predictive maintenance model...")
def get_service() -> PredictionService:
    svc = PredictionService(MODEL_DIR)
    if not svc.ready:
        from maintaina.data import load_ai4i
        from maintaina.train import train
        train(load_ai4i(), MODEL_DIR, n_estimators=int(os.getenv("TRAIN_TREES", "150")))
        svc.load()
    return svc


@st.cache_resource
def get_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.ingest_dir(KNOWLEDGE_DIR)
    return kb


def run_prediction(payload: dict) -> dict:
    if API_URL:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    result = get_service().predict(payload)
    if result["alert"] or result["anomaly"]:
        result["guidance"] = guidance(
            get_kb(),
            build_query(result["unusual_inputs"], result["alert"], result["anomaly"]),
        )
    return result


def get_guidance(query: str) -> dict:
    if API_URL:
        response = requests.post(f"{API_URL}/guidance", json={"query": query}, timeout=60)
        response.raise_for_status()
        return response.json()
    return guidance(get_kb(), query)


if "history" not in st.session_state:
    st.session_state.history = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = None

# Keep the original startup check: in-process mode needs a loaded model.
if not API_URL:
    try:
        get_service()
    except Exception as exc:
        st.error(
            "The model is not available and could not be trained automatically "
            f"(dataset download failed?): {exc}\n\nRun `python -m maintaina.train` once, "
            "or place `ai4i2020.csv` in the `data/` folder."
        )
        st.stop()

# ---------- Navigation ----------
with st.sidebar:
    st.divider()
    page = st.radio(
        "WORKSPACE",
        [
            "Overview",
            "Predictive maintenance",
            "Maintenance copilot",
            "Analytics",
            "Knowledge base",
            "Model insights",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.caption("SYSTEM STATUS")
    st.markdown(
        f'<span class="pill">{"API connected" if API_URL else "Local model mode"}</span>',
        unsafe_allow_html=True,
    )
    st.caption("Decision-support only · Always follow qualified inspection procedures.")

# ---------- Reusable page header ----------
def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result: dict) -> None:
    score = float(result.get("failure_probability", 0))
    threshold = float(result.get("threshold", 0.34))
    c1, c2, c3 = st.columns(3)
    c1.metric("Model failure score", f"{score:.1%}")
    c2.metric("Alert threshold", f"{threshold:.0%}")
    c3.metric("Unusual sensor pattern", "Detected" if result.get("anomaly") else "Not detected")

    if result.get("alert"):
        st.error("Risk score is above the alert threshold. Consider a qualified inspection.")
    else:
        st.success("Risk score is below the alert threshold. This does not guarantee that the machine is healthy.")

    if result.get("anomaly"):
        st.warning(
            "This sensor combination is unusual compared with training data. "
            "An anomaly is not the same as a machine failure."
        )
    for warning in result.get("warnings", []):
        st.warning(warning)

    unusual = result.get("unusual_inputs", [])
    if unusual:
        st.markdown('<div class="section-title">Inputs outside typical ranges</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(unusual), hide_index=True, use_container_width=True)

    retrieved = result.get("guidance")
    if retrieved and retrieved.get("passages"):
        st.markdown('<div class="section-title">Retrieved maintenance evidence</div>', unsafe_allow_html=True)
        st.caption(retrieved.get("message", "Quoted passages from your documents."))
        for passage in retrieved["passages"]:
            source_label = passage.get("source", "Document")
            if passage.get("page"):
                source_label += f" · Page {passage['page']}"
            with st.expander(f"{source_label}  |  relevance {passage.get('score', '—')}"):
                st.write(passage.get("text", ""))

    st.caption(DISCLAIMER)


# ---------- Overview ----------
if page == "Overview":
    page_header(
        "MACHINE INTELLIGENCE / OVERVIEW",
        "Machine health at a glance",
        "Monitor sensor readings, review risk assessments and explore maintenance evidence.",
    )

    history = st.session_state.history
    latest = st.session_state.latest_result
    total = len(history)
    alerts = sum(1 for item in history if item["alert"])
    anomalies = sum(1 for item in history if item["anomaly"])
    latest_score = f"{latest['failure_probability']:.1%}" if latest else "—"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assessments this session", total)
    k2.metric("Above-threshold alerts", alerts)
    k3.metric("Unusual patterns", anomalies)
    k4.metric("Latest failure score", latest_score)

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="section-title">Recent assessments</div>', unsafe_allow_html=True)
        if history:
            recent = pd.DataFrame(history[-8:])
            recent = recent.rename(
                columns={"score": "Failure score", "type": "Type", "alert": "Alert", "anomaly": "Anomaly"}
            )
            st.dataframe(
                recent[["Type", "Failure score", "Alert", "Anomaly"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No assessments yet. Open Predictive maintenance to assess a machine reading.")
    with right:
        st.markdown('<div class="section-title">System snapshot</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Prediction engine</div>
                <div class="info-value">{"Online" if (API_URL or get_service().ready) else "Unavailable"}</div>
                <div class="info-note">{"FastAPI service" if API_URL else "Local model pipeline"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">Knowledge documents</div>
                <div class="info-value">{len(list(KNOWLEDGE_DIR.glob('*'))) if KNOWLEDGE_DIR.exists() else 0}</div>
                <div class="info-note">Files in the local knowledge folder</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-title">How Maintaina AI works</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown("**01 · Read sensors**")
        st.caption("Enter operating conditions such as temperature, speed, torque and tool wear.")
    with b:
        st.markdown("**02 · Assess risk**")
        st.caption("The trained pipeline estimates a failure score and checks for unusual patterns.")
    with c:
        st.markdown("**03 · Review evidence**")
        st.caption("The copilot retrieves relevant passages from the maintenance documents you provide.")

# ---------- Predictive maintenance ----------
elif page == "Predictive maintenance":
    page_header(
        "PREDICT / ASSESSMENT",
        "Predictive maintenance",
        "Submit a machine sensor snapshot to estimate risk and identify unusual operating conditions.",
    )
    with st.form("machine_reading_form"):
        st.markdown("**Machine profile**")
        c1, c2, c3 = st.columns(3)
        with c1:
            mtype = st.selectbox("Product type", ["L", "M", "H"], index=1)
            air = st.number_input("Air temperature (K)", 250.0, 400.0, 300.0, 0.1)
        with c2:
            proc = st.number_input("Process temperature (K)", 250.0, 450.0, 310.0, 0.1)
            rpm = st.number_input("Rotational speed (rpm)", 1.0, 10000.0, 1500.0, 10.0)
        with c3:
            torque = st.number_input("Torque (Nm)", 0.0, 500.0, 40.0, 0.5)
            wear = st.number_input("Tool wear (min)", 0.0, 1000.0, 100.0, 1.0)
        submitted = st.form_submit_button("⚡ Assess machine risk", type="primary", use_container_width=True)

    if submitted:
        try:
            payload = SensorReading(
                type=mtype,
                air_temperature=air,
                process_temperature=proc,
                rotational_speed=rpm,
                torque=torque,
                tool_wear=wear,
            ).model_dump()
            with st.spinner("Evaluating sensor readings..."):
                result = run_prediction(payload)
            st.session_state.latest_result = result
            st.session_state.latest_reading = payload
            st.session_state.history.append(
                {
                    "type": mtype,
                    "air": air,
                    "proc": proc,
                    "rpm": rpm,
                    "torque": torque,
                    "wear": wear,
                    "score": result["failure_probability"],
                    "alert": result["alert"],
                    "anomaly": result["anomaly"],
                }
            )
            st.success("Assessment completed.")
        except Exception as exc:
            st.error(f"Could not assess this reading: {exc}")

    if st.session_state.latest_result:
        st.markdown('<div class="section-title">Latest assessment</div>', unsafe_allow_html=True)
        render_result(st.session_state.latest_result)
    else:
        st.info("Enter sensor values above and select Assess machine risk to see the result.")

# ---------- Maintenance copilot ----------
elif page == "Maintenance copilot":
    page_header(
        "KNOWLEDGE / COPILOT",
        "Maintenance copilot",
        "Search your uploaded manuals and retrieve relevant passages with document and page references.",
    )
    st.info("The copilot retrieves evidence from your documents. It does not create or replace approved maintenance procedures.")
    query = st.text_area(
        "What would you like to find in the manuals?",
        placeholder="Example: safe isolation steps before inspecting industrial machinery",
        height=100,
    )
    if st.button("Search maintenance documents", type="primary"):
        if not query.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching your knowledge base..."):
                    answer = get_guidance(query.strip())
                passages = answer.get("passages", [])
                if passages:
                    st.success(f"Found {len(passages)} relevant passage(s).")
                    st.caption(answer.get("message", "Retrieved document evidence."))
                    for passage in passages:
                        label = passage.get("source", "Document")
                        if passage.get("page"):
                            label += f" · Page {passage['page']}"
                        with st.expander(f"{label} · relevance {passage.get('score', '—')}", expanded=True):
                            st.write(passage.get("text", ""))
                else:
                    st.warning(answer.get("message", "No sufficiently relevant passage was found in your documents."))
            except Exception as exc:
                st.error(f"Could not search the knowledge base: {exc}")

# ---------- Analytics ----------
elif page == "Analytics":
    page_header(
        "INSIGHTS / SESSION ANALYTICS",
        "Assessment analytics",
        "Explore the predictions made during this browser session. These are snapshots, not a machine time series.",
    )
    if not st.session_state.history:
        st.info("Analytics will appear after you run a few assessments.")
    else:
        df = pd.DataFrame(st.session_state.history)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total assessments", len(df))
        c2.metric("Mean model score", f"{df['score'].mean():.1%}")
        c3.metric("Alert rate", f"{df['alert'].mean():.1%}")
        chart_data = df[["score"]].copy()
        chart_data.index = [f"Run {i + 1}" for i in range(len(chart_data))]
        st.markdown('<div class="section-title">Failure score by assessment</div>', unsafe_allow_html=True)
        st.line_chart(chart_data, use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.markdown("**Alert outcomes**")
            outcome = df["alert"].value_counts().rename(index={True: "Above threshold", False: "Below threshold"})
            st.bar_chart(outcome)
        with right:
            st.markdown("**Anomaly checks**")
            anomaly_counts = df["anomaly"].value_counts().rename(index={True: "Unusual", False: "Not unusual"})
            st.bar_chart(anomaly_counts)
        st.markdown("**Full assessment log**")
        st.dataframe(df, hide_index=True, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download session data (CSV)", csv, "maintaina_session_history.csv", "text/csv")

# ---------- Knowledge base ----------
elif page == "Knowledge base":
    page_header(
        "DOCUMENTS / KNOWLEDGE BASE",
        "Maintenance knowledge base",
        "Add equipment manuals and maintenance guidance documents for retrieval by the copilot.",
    )
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])
    if uploaded is not None:
        if st.button("Add document to knowledge base", type="primary"):
            safe_name = Path(uploaded.name).name
            (KNOWLEDGE_DIR / safe_name).write_bytes(uploaded.getvalue())
            st.cache_resource.clear()
            st.success(f"Added {safe_name}. The knowledge base will be reloaded on the next search.")
    documents = sorted(
        [p for p in KNOWLEDGE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}],
        key=lambda p: p.name.lower(),
    )
    st.markdown('<div class="section-title">Available documents</div>', unsafe_allow_html=True)
    if documents:
        doc_rows = [{"Document": p.name, "Format": p.suffix.upper().replace(".", ""), "Size": f"{p.stat().st_size / 1024:.1f} KB"} for p in documents]
        st.dataframe(pd.DataFrame(doc_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No supported documents found yet. Upload a PDF, TXT or Markdown file to get started.")
    if st.button("Reload / re-index knowledge base"):
        try:
            st.cache_resource.clear()
            kb = get_kb()
            st.success(f"Knowledge base reloaded. Retrieved {len(kb.chunks)} chunks.")
        except Exception as exc:
            st.error(f"Could not re-index documents: {exc}")

# ---------- Model insights ----------
elif page == "Model insights":
    page_header(
        "TRANSPARENCY / MODEL",
        "Model insights",
        "Understand what the model reports, how the alert threshold is used and where the system has limitations.",
    )
    if not API_URL:
        try:
            metadata = get_service().meta
            st.markdown('<div class="section-title">Training metadata</div>', unsafe_allow_html=True)
            st.json(metadata)
        except Exception as exc:
            st.warning(f"Could not load model metadata: {exc}")
    else:
        st.info("The dashboard is connected to FastAPI. Local training metadata is not displayed in API mode.")
    st.markdown('<div class="section-title">Important limitations</div>', unsafe_allow_html=True)
    st.warning(
        "The model was trained using the AI4I 2020 dataset. Its scores are not calibrated guarantees "
        "of real-world failure probability. Validate it with representative machine data before operational use."
    )
    st.markdown(
        """
        - **Failure score:** a model output used to rank the submitted sensor snapshot.
        - **Alert threshold:** the cutoff used by the application to raise an alert.
        - **Anomaly flag:** indicates an unusual combination compared with training data; it does not mean failure.
        - **Knowledge retrieval:** shows passages from the documents you supplied; it does not approve maintenance work.
        - **Data shape:** the dataset contains snapshots, not continuous machine time-series.
        """
    )
    st.caption(DISCLAIMER)

st.sidebar.caption("Maintaina AI · Predictive Maintenance Copilot")
