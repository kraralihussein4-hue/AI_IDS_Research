
from pathlib import Path
import json
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.predict import predict_traffic, load_model

EXP = ROOT / "experiments" / "random_forest_binary"

METRICS_FILE = EXP / "metrics.json"
ATTACK_FILE = EXP / "attack_type_detection.csv"
CONFUSION_FILE = EXP / "confusion_matrix.csv"
IMPORTANCE_FILE = EXP / "feature_importance.csv"


# -----------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------

st.set_page_config(
    page_title="AI-IDS | Security Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------
# THEME
# -----------------------------------------------------

BG = "#080f1d"
PANEL = "#111d30"
BORDER = "#243650"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
BLUE = "#38bdf8"
GREEN = "#10b981"
RED = "#f87171"
AMBER = "#fbbf24"


# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stHeader"] {{
        background: {BG};
    }}

    [data-testid="stSidebar"] {{
        background: #0d1728;
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
    }}

    p, label {{
        color: {MUTED};
    }}

    .hero {{
        background: linear-gradient(
            120deg,
            #142944,
            #102033,
            #0c192b
        );
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 25px 28px;
        margin-bottom: 25px;
    }}

    .hero-title {{
        font-size: 27px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        color: #a8bad0;
        font-size: 14px;
    }}

    .section-heading {{
        font-size: 19px;
        font-weight: 750;
        color: #f1f5f9;
        margin: 25px 0 15px 0;
    }}

    .metric-card {{
        background: linear-gradient(
            145deg,
            #17263c,
            #101b2d
        );
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 18px;
        min-height: 125px;
    }}

    .metric-label {{
        color: #9fb0c5;
        font-size: 12px;
        font-weight: 650;
        letter-spacing: 0.5px;
    }}

    .metric-value {{
        color: #ffffff;
        font-size: 28px;
        font-weight: 800;
        margin-top: 13px;
    }}

    .metric-note {{
        color: #91a4ba;
        font-size: 11px;
        margin-top: 5px;
    }}

    .status-pill {{
        display: inline-block;
        background: #073b32;
        color: #34d399;
        border: 1px solid #12664f;
        border-radius: 20px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
    }}

    .info-panel {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 18px;
    }}

    .stButton button,
    .stDownloadButton button {{
        background: #123b61;
        color: white;
        border: 1px solid #28618c;
        border-radius: 8px;
    }}

    .stButton button:hover,
    .stDownloadButton button:hover {{
        background: #18517f;
        color: white;
        border-color: {BLUE};
    }}

    [data-testid="stMetricValue"] {{
        color: white;
    }}

    [data-testid="stMetricLabel"] {{
        color: #a8bad0;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------
# DATA LOADING
# -----------------------------------------------------

@st.cache_data
def load_json(path):

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


@st.cache_data
def load_confusion(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, index_col=0)


metrics = load_json(METRICS_FILE)
attack_data = load_csv(ATTACK_FILE)
importance_data = load_csv(IMPORTANCE_FILE)
confusion = load_confusion(CONFUSION_FILE)


# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------

def metric_card(label, value, note="", color=BLUE):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">
                {value}
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title):

    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True,
    )


def display_attack_name(name):

    return str(name).replace("�", "-")


def styled_chart(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            color=TEXT,
            family="Arial",
            size=12,
        ),
        margin=dict(l=20, r=20, t=35, b=25),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)"
        ),
    )

    fig.update_xaxes(
        gridcolor="#26374d",
        zerolinecolor="#26374d",
    )

    fig.update_yaxes(
        gridcolor="#26374d",
        zerolinecolor="#26374d",
    )

    return fig


# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div style="padding:18px 5px;text-align:center;">
            <div style="font-size:43px;">🛡️</div>
            <div style="font-size:23px;font-weight:850;">
                AI-IDS
            </div>
            <div style="font-size:10px;color:#8da4be;
                        letter-spacing:2px;">
                SECURITY CENTER
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("**NAVIGATION**")

    page = st.radio(
        "Select Module",
        [
            "Dashboard",
            "AI Detection",
            "Attack Analysis",
            "IDS Comparison",
            "Reports",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**SYSTEM STATUS**")

    if metrics:
        st.markdown(
            '<span class="status-pill">● MODEL AVAILABLE</span>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Model metrics unavailable")

    st.markdown(
        """
        <div style="margin-top:25px;
                    color:#91a4ba;font-size:12px;
                    line-height:2;">
            <b>MODEL</b><br>
            Random Forest<br><br>
            <b>DATASET</b><br>
            CICIDS2017<br><br>
            <b>CLASSIFICATION</b><br>
            Binary Classification<br><br>
            <b>FEATURES</b><br>
            78 Network Features
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------
# HEADER
# -----------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            AI-Based Intrusion Detection System
        </div>
        <div class="hero-subtitle">
            Intelligent Network Traffic Analysis |
            Machine Learning Threat Detection
        </div>
        <br>
        <span class="status-pill">
            ● RESEARCH MODEL AVAILABLE
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    section("Security Overview")

    if metrics is None:

        st.error("Model evaluation metrics are unavailable.")

    else:

        tp = fp = fn = tn = 0

        if not confusion.empty:
            try:
                tn = int(confusion.loc[
                    "Actual_BENIGN", "Predicted_BENIGN"
                ])
                fp = int(confusion.loc[
                    "Actual_BENIGN", "Predicted_ATTACK"
                ])
                fn = int(confusion.loc[
                    "Actual_ATTACK", "Predicted_BENIGN"
                ])
                tp = int(confusion.loc[
                    "Actual_ATTACK", "Predicted_ATTACK"
                ])
            except (KeyError, ValueError, TypeError):
                st.warning("Confusion matrix format is unexpected.")

        cards = st.columns(4)

        with cards[0]:
            metric_card(
                "MODEL ACCURACY",
                f"{metrics['accuracy']:.2%}",
                "Overall correct predictions",
                BLUE,
            )

        with cards[1]:
            metric_card(
                "ATTACK RECALL",
                f"{metrics['recall_attack']:.2%}",
                "Attacks successfully detected",
                GREEN,
            )

        with cards[2]:
            metric_card(
                "ATTACK PRECISION",
                f"{metrics['precision_attack']:.2%}",
                "Reliability of attack alerts",
                BLUE,
            )

        with cards[3]:
            metric_card(
                "F1 SCORE",
                f"{metrics['f1_attack']:.2%}",
                "Attack classification balance",
                GREEN,
            )

        section("Network Dataset Overview")

        stats = st.columns(3)

        stats[0].metric(
            "Training Records",
            f"{metrics['training_rows']:,}",
        )

        stats[1].metric(
            "Testing Records",
            f"{metrics['testing_rows']:,}",
        )

        stats[2].metric(
            "Network Features",
            metrics["features"],
        )

        section("Detection Activity")

        activity = st.columns(4)

        activity[0].metric("Detected Attacks", f"{tp:,}")
        activity[1].metric("Missed Attacks", f"{fn:,}")
        activity[2].metric("False Alerts", f"{fp:,}")
        activity[3].metric("Correct Benign", f"{tn:,}")

        if not confusion.empty:

            section("Classification Confusion Matrix")

            fig = px.imshow(
                confusion,
                text_auto=True,
                color_continuous_scale="Blues",
                aspect="auto",
                labels={
                    "x": "Predicted Class",
                    "y": "Actual Class",
                    "color": "Records",
                },
            )

            st.plotly_chart(
                styled_chart(fig, 350),
                use_container_width=True,
            )

        if not attack_data.empty:

            section("Attack Detection Performance")

            chart_data = attack_data.copy()

            chart_data["Display_Name"] = (
                chart_data["Attack_Type"].apply(
                    display_attack_name
                )
            )

            chart_data = chart_data.sort_values(
                "Detection_Rate",
                ascending=True,
            )

            fig = px.bar(
                chart_data,
                x="Detection_Rate",
                y="Display_Name",
                orientation="h",
                color="Detection_Rate",
                color_continuous_scale=[
                    RED,
                    AMBER,
                    GREEN,
                ],
                hover_data=[
                    "Total_Records",
                    "Detected",
                    "Missed",
                ],
                labels={
                    "Detection_Rate": "Detection Rate",
                    "Display_Name": "Attack Type",
                },
            )

            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(coloraxis_showscale=False)

            st.plotly_chart(
                styled_chart(fig, 540),
                use_container_width=True,
            )


# =====================================================
# AI DETECTION - REAL PREDICTION
# =====================================================

elif page == "AI Detection":

    section("Artificial Intelligence Detection Engine")

    st.markdown(
        """
        <div class="info-panel">
            <h3>Random Forest Detection Engine</h3>
            <p>
                Upload a network traffic CSV file.
                The trained model will classify each record
                as BENIGN or ATTACK.
            </p>
            <p>
                Required input: 78 CICIDS2017 network features.
                Dataset label columns are optional and ignored.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics:

        cols = st.columns(3)

        cols[0].metric(
            "Model ROC-AUC",
            f"{metrics['roc_auc']:.2%}",
        )

        cols[1].metric(
            "Balanced Accuracy",
            f"{metrics['balanced_accuracy']:.2%}",
        )

        cols[2].metric(
            "Training Records",
            f"{metrics['training_rows']:,}",
        )

    section("Download CSV Template")

    @st.cache_resource
    def get_model_feature_names():
        model = load_model()
        if not hasattr(model, "feature_names_in_"):
            raise ValueError(
                "The saved model does not expose feature_names_in_. "
                "Please verify the trained model."
            )
        return list(model.feature_names_in_)

    try:
        model_feature_names = get_model_feature_names()

        template_data = pd.DataFrame(
            [{feature: 0 for feature in model_feature_names}]
        )

        st.download_button(
            label="Download CSV Template (78 Features)",
            data=template_data.to_csv(index=False).encode("utf-8"),
            file_name="AI_IDS_78_Features_Template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Downloads a CSV containing the exact feature names expected by the saved model.",
        )

        st.info(
            "Download this template, replace the placeholder zeros with "
            "real network-traffic feature values, then upload the completed "
            "CSV below. The zeros are placeholders and are not real traffic."
        )

    except Exception as error:
        st.error(f"Could not create the CSV template: {error}")

    # =====================================================
    # DOWNLOAD REAL CICIDS2017 SAMPLE DATASETS
    # =====================================================

    section("Download Sample Datasets")

    st.markdown(
        """
        Download real CICIDS2017 network traffic samples
        and upload them below to test the AI detection model.
        The Label column is included for reference and is ignored
        automatically by the prediction pipeline.
        """
    )

    sample_dir = ROOT / "dashboard" / "sample_data"

    sample_files = [
        ("BENIGN.csv", "Download BENIGN Sample"),
        ("ATTACK.csv", "Download ATTACK Sample"),
        ("MIXED.csv", "Download MIXED Sample"),
    ]

    sample_cols = st.columns(3)

    for col, (filename, button_label) in zip(sample_cols, sample_files):
        file_path = sample_dir / filename

        with col:
            if file_path.is_file():
                st.download_button(
                    label=button_label,
                    data=file_path.read_bytes(),
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_sample_{filename}",
                )
            else:
                st.warning(f"{filename} is unavailable.")

    section("Upload Network Traffic")

    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file",
        type=["csv"],
        help="The CSV must contain all 78 model input features.",
        key="traffic_csv_uploader",
    )

    if uploaded_file is not None:

        st.success(
            f"File uploaded: {uploaded_file.name}"
        )

        try:

            uploaded_data = pd.read_csv(
                uploaded_file,
                low_memory=False,
            )

            if uploaded_data.empty:
                st.error("The uploaded CSV file contains no data rows.")
                st.stop()

            st.write(
                f"Records: {len(uploaded_data):,} | "
                f"Columns: {len(uploaded_data.columns):,}"
            )

            st.caption(
                "The file is analyzed in memory. "
                "It is not saved or uploaded to an external service."
            )

            if len(uploaded_data) > 500_000:

                st.warning(
                    "This file contains more than 500,000 records. "
                    "Processing may require significant memory and time."
                )

            if st.button(
                "Run AI Threat Detection",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Random Forest is analyzing network traffic..."
                ):

                    try:

                        results = predict_traffic(
                            uploaded_data
                        )

                        results.insert(
                            0,
                            "Record_ID",
                            range(1, len(results) + 1),
                        )

                        attack_count = int(
                            (results["Prediction"] == "ATTACK").sum()
                        )

                        benign_count = int(
                            (results["Prediction"] == "BENIGN").sum()
                        )

                        total_count = len(results)

                        attack_rate = (
                            attack_count / total_count
                            if total_count else 0
                        )

                        st.session_state[
                            "detection_results"
                        ] = results

                        st.session_state[
                            "detection_filename"
                        ] = uploaded_file.name

                        st.success(
                            "Network traffic analysis completed."
                        )

                    except (ValueError, TypeError, FileNotFoundError) as error:

                        st.error(
                            f"Detection could not be completed: {error}"
                        )

                    except Exception as error:

                        st.error(
                            f"Unexpected detection error: {error}"
                        )

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:

            st.error(
                f"Could not read the uploaded CSV file: {error}"
            )

    # -------------------------------------------------
    # DISPLAY STORED RESULTS
    # -------------------------------------------------

    results = st.session_state.get(
        "detection_results"
    )

    if results is not None:

        section("Detection Results")

        total_count = len(results)

        attack_count = int(
            (results["Prediction"] == "ATTACK").sum()
        )

        benign_count = int(
            (results["Prediction"] == "BENIGN").sum()
        )

        attack_rate = (
            attack_count / total_count
            if total_count else 0
        )

        result_cards = st.columns(4)

        result_cards[0].metric(
            "Total Records",
            f"{total_count:,}",
        )

        result_cards[1].metric(
            "Detected Attacks",
            f"{attack_count:,}",
        )

        result_cards[2].metric(
            "Benign Connections",
            f"{benign_count:,}",
        )

        result_cards[3].metric(
            "Attack Percentage",
            f"{attack_rate:.2%}",
        )

        if attack_count > 0:

            st.error(
                f"Potential threats detected: {attack_count:,}"
            )

        else:

            st.success(
                "No records were classified as attacks."
            )

        section("Traffic Classification Distribution")

        distribution = pd.DataFrame({
            "Classification": [
                "BENIGN",
                "ATTACK",
            ],
            "Records": [
                benign_count,
                attack_count,
            ],
        })

        fig = px.pie(
            distribution,
            names="Classification",
            values="Records",
            hole=0.58,
            color="Classification",
            color_discrete_map={
                "BENIGN": GREEN,
                "ATTACK": RED,
            },
        )

        fig.update_traces(
            textposition="inside",
            textinfo="label+percent",
        )

        st.plotly_chart(
            styled_chart(fig, 390),
            use_container_width=True,
        )

        section("Individual Connection Results")

        show_only_attacks = st.checkbox(
            "Show only detected attacks",
            value=False,
        )

        display_results = results.copy()

        display_results["Attack_Probability"] = (
            display_results["Attack_Probability"] * 100
        )

        display_results["Benign_Probability"] = (
            display_results["Benign_Probability"] * 100
        )

        display_results["Confidence"] = (
            display_results["Confidence"] * 100
        )

        if show_only_attacks:

            display_results = display_results[
                display_results["Prediction"] == "ATTACK"
            ]

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Attack_Probability": st.column_config.ProgressColumn(
                    "Attack Probability (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Benign_Probability": st.column_config.NumberColumn(
                    "Benign Probability (%)",
                    format="%.2f%%",
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence (%)",
                    format="%.2f%%",
                ),
            },
        )

        csv_output = results.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Download Detection Results (CSV)",
            data=csv_output,
            file_name="ai_ids_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "Probabilities are model estimates, not guarantees "
            "that a connection is malicious or safe."
        )

    section("Most Important Network Features")

    if not importance_data.empty:

        top_features = importance_data.head(10).sort_values(
            "Importance",
            ascending=True,
        )

        fig = px.bar(
            top_features,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=[
                "#1d4ed8",
                BLUE,
            ],
        )

        fig.update_layout(
            coloraxis_showscale=False
        )

        st.plotly_chart(
            styled_chart(fig, 430),
            use_container_width=True,
        )


# =====================================================
# ATTACK ANALYSIS
# =====================================================

elif page == "Attack Analysis":

    section("Attack Classification Analysis")

    if attack_data.empty:

        st.warning("Attack report is unavailable.")

    else:

        display_data = attack_data.copy()

        display_data["Attack_Type"] = (
            display_data["Attack_Type"].apply(
                display_attack_name
            )
        )

        display_data["Detection_Rate"] = (
            display_data["Detection_Rate"] * 100
        )

        display_data = display_data.rename(
            columns={
                "Attack_Type": "Attack Type",
                "Total_Records": "Total Records",
                "Detected": "Detected",
                "Missed": "Missed",
                "Detection_Rate": "Detection Rate (%)",
            }
        )

        styled_table = (
            display_data.style
            .format({
                "Detection Rate (%)": "{:.2f}"
            })
            .set_properties(**{
                "background-color": "#111d30",
                "color": "#f1f5f9",
                "border-color": "#243650",
            })
            .set_table_styles([
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#1b2d45"),
                        ("color", "#38bdf8"),
                        ("font-weight", "bold"),
                        ("border-color", "#243650"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-color", "#243650"),
                    ],
                },
            ])
        )

        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True,
        )

        section("Individual Attack Inspection")

        selected = st.selectbox(
            "Choose attack type",
            display_data["Attack Type"].tolist(),
        )

        row = display_data[
            display_data["Attack Type"] == selected
        ].iloc[0]

        cols = st.columns(4)

        cols[0].metric(
            "Total",
            f"{int(row['Total Records']):,}"
        )

        cols[1].metric(
            "Detected",
            f"{int(row['Detected']):,}"
        )

        cols[2].metric(
            "Missed",
            f"{int(row['Missed']):,}"
        )

        cols[3].metric(
            "Detection Rate",
            f"{row['Detection Rate (%)']:.2f}%"
        )


# =====================================================
# IDS COMPARISON
# =====================================================

elif page == "IDS Comparison":

    section("AI vs Traditional IDS Comparative Evaluation")

    comparison_file = ROOT / "reports" / "ids_comparison.json"

    if not comparison_file.exists():
        st.error("Comparison results file is unavailable.")
    else:
        with open(comparison_file, "r", encoding="utf-8") as file:
            comparison = json.load(file)

        rf = comparison["random_forest"]
        snort = comparison["snort"]
        suricata = comparison["suricata"]

        section("Random Forest Performance")

        cols = st.columns(4)

        cols[0].metric("Accuracy", f"{rf['accuracy']:.2%}")
        cols[1].metric("Precision", f"{rf['precision']:.2%}")
        cols[2].metric("Recall", f"{rf['recall']:.2%}")
        cols[3].metric("F1-Score", f"{rf['f1_score']:.2%}")

        cols = st.columns(3)
        cols[0].metric("ROC-AUC", f"{rf['roc_auc']:.2%}")
        cols[1].metric("Testing Records", f"{rf['testing_rows']:,}")
        cols[2].metric("Features", rf["features"])

        section("Traditional IDS Functional Test")

        cols = st.columns(2)

        with cols[0]:
            st.markdown("### Snort")
            st.metric("Version", snort["version"])
            st.metric("Packets Analyzed", snort["packets_analyzed"])
            st.metric("Alerts Generated", snort["alerts"])
            st.metric("Runtime", f"{snort['runtime_seconds']:.6f} sec")

        with cols[1]:
            st.markdown("### Suricata")
            st.metric("Version", suricata["version"])
            st.metric("Packets Analyzed", suricata["packets_analyzed"])
            st.metric("Alerts Generated", suricata["alerts"])
            runtime = suricata["runtime_seconds"]
            st.metric("Runtime", f"{runtime:.6f} sec" if runtime is not None else "Not measured")

        section("Traditional IDS Alert Comparison")

        alert_data = pd.DataFrame({
            "IDS Engine": ["Snort", "Suricata"],
            "Alerts": [snort["alerts"], suricata["alerts"]],
            "Packets Analyzed": [
                snort["packets_analyzed"],
                suricata["packets_analyzed"]
            ]
        })

        fig = px.bar(
            alert_data,
            x="IDS Engine",
            y="Alerts",
            color="IDS Engine",
            text="Alerts",
            title="Alerts Generated on the Same Synthetic PCAP"
        )

        fig.update_traces(textposition="outside")
        st.plotly_chart(
            styled_chart(fig, 350),
            use_container_width=True
        )

        st.dataframe(
            alert_data,
            use_container_width=True,
            hide_index=True
        )

        section("Experimental Methodology")

        st.info(comparison["methodology_note"])

        st.caption(
            "The Random Forest evaluation uses a held-out dataset. "
            "Snort and Suricata were tested functionally using the same "
            "five-packet synthetic PCAP. Their metrics are not directly "
            "comparable."
        )

# =====================================================
# =====================================================
# REPORTS
# =====================================================

elif page == "IDS Comparison":

    section("Traditional IDS Comparative Evaluation")

    st.warning(
        "Snort and Suricata experiments have not yet been "
        "performed. No comparison values are displayed."
    )

    st.markdown(
        """
        <div class="info-panel">
            <h3>Planned Comparative Metrics</h3>
            <p>Detection Rate | False Positive Rate</p>
            <p>Precision | Recall | F1-Score</p>
            <p>Processing Time | Resource Usage</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "AI, Snort and Suricata will be compared using "
        "a consistent experimental methodology."
    )


# =====================================================
# REPORTS
# =====================================================

elif page == "Reports":

    section("Research Reports & Downloads")

    if metrics:

        st.download_button(
            "Download Model Metrics (JSON)",
            data=json.dumps(metrics, indent=4),
            file_name="random_forest_metrics.json",
            mime="application/json",
            use_container_width=True,
        )

    if not attack_data.empty:

        st.download_button(
            "Download Attack Analysis (CSV)",
            data=attack_data.to_csv(index=False),
            file_name="attack_type_detection.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not confusion.empty:

        st.download_button(
            "Download Confusion Matrix (CSV)",
            data=confusion.to_csv(),
            file_name="confusion_matrix.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not importance_data.empty:

        st.download_button(
            "Download Feature Importance (CSV)",
            data=importance_data.to_csv(index=False),
            file_name="feature_importance.csv",
            mime="text/csv",
            use_container_width=True,
        )

    detection_results = st.session_state.get(
        "detection_results"
    )

    if detection_results is not None:

        st.download_button(
            "Download Latest AI Detection Results",
            data=detection_results.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="ai_ids_detection_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#71849d;
        font-size:12px;
        padding:10px;">
        AI-IDS Research Project |
        CICIDS2017 |
        Random Forest
    </div>
    """,
    unsafe_allow_html=True,
)
