"""
AI Predictive Maintenance Dashboard.

All ML inference is performed by the existing FastAPI backend.
This file is only responsible for presentation and API communication.
"""

import os
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from api_client import (
    APIClientError,
    DEFAULT_API_URL,
    PredictiveMaintenanceAPI,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #101722;
        color: #E7EDF5;
    }

    [data-testid="stSidebar"] {
        background-color: #172333;
    }

    [data-testid="stMetric"] {
        background-color: #1B2A3B;
        border: 1px solid #30445B;
        padding: 16px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #AAB8C8;
    }

    [data-testid="stMetricValue"] {
        color: #F2F6FC;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #147D92;
        color: white;
        border-radius: 8px;
        border: none;
        min-height: 42px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #189BB4;
        color: white;
        border: none;
    }

    .dashboard-heading {
        font-size: 30px;
        font-weight: 700;
        color: #F2F6FC;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        color: #AAB8C8;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .section-heading {
        color: #65C7D7;
        font-size: 20px;
        font-weight: 650;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "single_result": None,
    "batch_result": None,
    "batch_filename": None,
    "batch_file_bytes": None,
    "prediction_download": None,
    "rejected_download": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def display_value(value, default="Not available"):
    if value is None:
        return default

    if isinstance(value, str) and not value.strip():
        return default

    return value


def parse_numeric_input(value):
    try:
        number = float(value)

        if not pd.notna(number):
            return None

        if number in (float("inf"), float("-inf")):
            return None

        return number

    except (TypeError, ValueError):
        return None


def display_top_factors(explanations):
    if not isinstance(explanations, dict):
        st.info("No explanation data was returned.")
        return

    st.write(
        "**Explanation method:**",
        display_value(explanations.get("explanation_method")),
    )

    factors = explanations.get("top_factors", [])

    if not factors:
        st.info("No top-factor explanation is available.")
        return

    if isinstance(factors, dict):
        factors = [
            {"feature": key, "value": value}
            for key, value in factors.items()
        ]

    if isinstance(factors, list):
        st.dataframe(
            pd.DataFrame(factors),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.json(factors)


def display_anomaly_details(anomaly_details):
    if not isinstance(anomaly_details, dict) or not anomaly_details:
        st.info("No individual detector results were returned.")
        return

    for detector, result in anomaly_details.items():
        if isinstance(result, dict):
            flag = result.get("is_anomaly", "Unknown")
        else:
            flag = result

        st.write(f"**{detector}:** {flag}")


# =========================================================
# SINGLE PREDICTION DISPLAY
# =========================================================

def display_prediction_result(result):
    if not isinstance(result, dict):
        st.error("The API returned an invalid prediction.")
        return

    if not result.get("success", True):
        st.error("The prediction was not successful.")
        st.json(result)
        return

    st.markdown(
        '<div class="section-heading">Equipment assessment</div>',
        unsafe_allow_html=True,
    )

    probability = result.get("failure_probability")

    if isinstance(probability, (float, int)):
        probability_text = f"{probability * 100:.2f}%"
    else:
        probability_text = "Not available"

    metadata = result.get("model_metadata") or {}

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Failure prediction",
        display_value(result.get("failure_prediction")),
    )

    col2.metric(
        "Failure probability",
        probability_text,
    )

    col3.metric(
        "Overall risk",
        display_value(result.get("risk_level")),
    )

    col4.metric(
        "Maintenance priority",
        display_value(metadata.get("maintenance_priority")),
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Maintenance recommendation")

        recommendations = result.get("recommendations", [])

        if recommendations:
            for item in recommendations:
                st.write(f"• {item}")
        else:
            st.info("No recommendation was returned.")

        st.markdown("#### Prediction metadata")

        st.write(
            "**Equipment ID:**",
            display_value(result.get("equipment_id")),
        )

        st.write(
            "**Model:**",
            display_value(metadata.get("model_name")),
        )

        st.write(
            "**Decision threshold:**",
            display_value(metadata.get("decision_threshold")),
        )

        st.write(
            "**Classifier risk:**",
            display_value(metadata.get("classifier_risk_level")),
        )

        st.write(
            "**Timestamp:**",
            display_value(result.get("timestamp")),
        )

    with right:
        st.markdown("#### Anomaly detection")

        st.write(
            "**Anomaly detected:**",
            display_value(result.get("anomaly_detected")),
        )

        display_anomaly_details(
            result.get("anomaly_details")
        )

    st.markdown("---")

    st.markdown("#### Explainable AI")

    display_top_factors(
        result.get("explanations")
    )

    st.markdown("---")

    st.markdown("#### Remaining useful life assessment")

    if result.get("rul_available", False):
        st.success("RUL assessment is available.")

        st.write(
            "**RUL prediction:**",
            display_value(result.get("rul_prediction")),
        )

        st.write(
            "**Risk before RUL:**",
            display_value(result.get("risk_level_before_rul")),
        )

        st.write(
            "**RUL risk level:**",
            display_value(result.get("rul_risk_level")),
        )

        st.write(
            "**RUL escalation applied:**",
            result.get("rul_escalation_applied", False),
        )
    else:
        st.info(
            "RUL is unavailable for this prediction. "
            "No separate compatible NASA C-MAPSS input was supplied."
        )

    health_score = metadata.get("health_score")

    if health_score is not None:
        st.markdown("---")
        st.markdown("#### Health score indicator")

        st.metric(
            "Returned health score",
            health_score,
        )

        st.caption(
            metadata.get(
                "health_score_note",
                "This is not a calibrated physical equipment-health measurement.",
            )
        )

    with st.expander("Complete prediction response"):
        st.json(result)


# =========================================================
# BATCH SUMMARY
# =========================================================

def display_batch_summary(result):
    summary = result.get("summary", {})

    if not isinstance(summary, dict):
        st.error("Invalid batch summary returned by API.")
        return

    st.markdown(
        '<div class="section-heading">Batch processing summary</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Input rows",
        summary.get("total_rows", 0),
    )

    col2.metric(
        "Successful predictions",
        summary.get("successful_predictions", 0),
    )

    col3.metric(
        "Rejected rows",
        summary.get("rejected_rows", 0),
    )

    col4.metric(
        "Failure predictions",
        summary.get("failure_predictions", 0),
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Normal predictions",
        summary.get("normal_predictions", 0),
    )

    col6.metric(
        "Majority-consensus anomalies",
        summary.get("rows_with_anomaly", 0),
    )

    col7.metric(
        "Any detector flagged",
        summary.get("rows_with_any_anomaly_detector", 0),
    )

    col8.metric(
        "SHAP explanations",
        summary.get("shap_explanations_available", 0),
    )

    st.caption(
        "Majority-consensus anomalies and any-detector flags "
        "represent different measurements."
    )

    risk_distribution = summary.get("risk_distribution", {})

    if isinstance(risk_distribution, dict) and risk_distribution:
        st.markdown("#### Risk distribution")

        chart_data = pd.DataFrame(
            [
                {
                    "Risk level": level,
                    "Equipment count": count,
                }
                for level, count in risk_distribution.items()
            ]
        )

        chart = px.bar(
            chart_data,
            x="Risk level",
            y="Equipment count",
            color="Risk level",
            text="Equipment count",
            template="plotly_dark",
        )

        chart.update_layout(
            paper_bgcolor="#101722",
            plot_bgcolor="#101722",
            font_color="#E7EDF5",
            showlegend=False,
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    with st.expander("Batch metadata"):
        st.json(summary)


# =========================================================
# BATCH TABLES
# =========================================================

def display_batch_tables(result):
    predictions = result.get("preview_rows", [])
    rejected = result.get("rejected_preview_rows", [])

    st.markdown("#### Successful prediction preview")

    if predictions:
        st.dataframe(
            pd.DataFrame(predictions),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No successful prediction rows were returned.")

    st.markdown("#### Rejected row preview")

    if rejected:
        st.dataframe(
            pd.DataFrame(rejected),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No rejected rows were returned.")


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("## ⚙️ Maintenance AI")

st.sidebar.caption(
    "Industrial equipment intelligence platform"
)

default_url = os.getenv(
    "PREDICTIVE_MAINTENANCE_API_URL",
    DEFAULT_API_URL,
)

api_url = st.sidebar.text_input(
    "FastAPI base URL",
    value=default_url,
)

api = PredictiveMaintenanceAPI(api_url)

st.sidebar.markdown("---")

if st.sidebar.button(
    "Check backend connection",
    use_container_width=True,
):
    try:
        health = api.health()

        st.sidebar.success(
            f"Connected: {health.get('status', 'Unknown')}"
        )

        st.sidebar.caption(
            health.get("service", "")
        )

    except APIClientError as exc:
        st.sidebar.error(str(exc))

st.sidebar.markdown("---")

st.sidebar.caption(
    "Frontend communicates with the existing FastAPI services."
)


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    '<div class="dashboard-heading">'
    'AI Predictive Maintenance System'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Industrial equipment monitoring · Failure prediction · '
    'Anomaly detection · Explainable AI'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# BACKEND STATUS
# =========================================================

status_col1, status_col2 = st.columns([1, 4])

try:
    health = api.health()

    with status_col1:
        st.success("API ONLINE")

    with status_col2:
        st.caption(
            f"{health.get('service', 'Predictive Maintenance API')} "
            f"· Version {health.get('version', 'Unknown')}"
        )

except APIClientError as exc:
    health = None

    with status_col1:
        st.error("API OFFLINE")

    with status_col2:
        st.warning(str(exc))

st.markdown("---")


# =========================================================
# MAIN TABS
# =========================================================

overview_tab, single_tab, batch_tab = st.tabs(
    [
        "📊 Overview",
        "🔍 Equipment Assessment",
        "📁 Batch Analysis",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

with overview_tab:

    st.markdown(
        '<div class="section-heading">System overview</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Monitor backend connectivity and review the latest "
        "equipment prediction results."
    )

    if health:
        col1, col2, col3 = st.columns(3)

        col1.metric("Backend status", "Online")
        col2.metric("API version", health.get("version", "Unknown"))
        col3.metric("Prediction service", "Available")
    else:
        st.warning(
            "The dashboard cannot currently reach the prediction API."
        )

    if st.session_state.batch_result:
        st.markdown("---")
        st.markdown("#### Latest batch assessment")

        display_batch_summary(
            st.session_state.batch_result
        )
    else:
        st.info(
            "No batch assessment has been performed in this session."
        )

    if st.session_state.single_result:
        st.markdown("---")
        st.markdown("#### Latest equipment assessment")

        result = st.session_state.single_result

        probability = result.get("failure_probability")

        if isinstance(probability, (float, int)):
            probability = f"{probability * 100:.2f}%"
        else:
            probability = "Not available"

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction",
            display_value(result.get("failure_prediction")),
        )

        col2.metric(
            "Failure probability",
            probability,
        )

        col3.metric(
            "Risk level",
            display_value(result.get("risk_level")),
        )


# =========================================================
# SINGLE EQUIPMENT ASSESSMENT
# =========================================================

with single_tab:

    st.markdown(
        '<div class="section-heading">Live equipment assessment</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Enter sensor readings and submit them to the registered "
        "prediction model through FastAPI."
    )

    with st.form("single_prediction_form"):

        equipment_id = st.text_input(
            "Equipment identifier (optional)",
            placeholder="MACHINE-001",
        )

        st.markdown("#### Temperature readings")

        col1, col2 = st.columns(2)

        with col1:
            air_temperature = st.text_input(
                "Air temperature [K]",
                placeholder="Enter value",
            )

        with col2:
            process_temperature = st.text_input(
                "Process temperature [K]",
                placeholder="Enter value",
            )

        st.markdown("#### Mechanical readings")

        col1, col2, col3 = st.columns(3)

        with col1:
            rotational_speed = st.text_input(
                "Rotational speed [rpm]",
                placeholder="Enter value",
            )

        with col2:
            torque = st.text_input(
                "Torque [Nm]",
                placeholder="Enter value",
            )

        with col3:
            tool_wear = st.text_input(
                "Tool wear [min]",
                placeholder="Enter value",
            )

        machine_type = st.selectbox(
            "Machine type",
            options=["L", "M", "H"],
        )

        submitted = st.form_submit_button(
            "Run equipment assessment",
            use_container_width=True,
        )

    if submitted:

        raw_values = {
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }

        sensor_data = {}
        invalid_fields = []

        for field, raw_value in raw_values.items():
            number = parse_numeric_input(raw_value)

            if number is None:
                invalid_fields.append(field)
            else:
                sensor_data[field] = number

        if invalid_fields:
            st.error(
                "Enter valid finite numeric values for: "
                + ", ".join(invalid_fields)
            )

        else:
            sensor_data["Type"] = machine_type

            payload = {
                "sensor_data": sensor_data,
                "equipment_id": equipment_id.strip() or None,
            }

            with st.spinner(
                "Running integrated equipment assessment..."
            ):
                try:
                    result = api.predict_single(payload)

                    st.session_state.single_result = result

                    st.success(
                        "Equipment assessment completed."
                    )

                except APIClientError as exc:
                    st.error(str(exc))

    if st.session_state.single_result:
        st.markdown("---")

        display_prediction_result(
            st.session_state.single_result
        )


# =========================================================
# CSV BATCH ANALYSIS
# =========================================================

with batch_tab:

    st.markdown(
        '<div class="section-heading">Batch equipment analysis</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Upload an AI4I-compatible CSV dataset. "
        "The backend validates individual rows and reports "
        "successful and rejected rows separately."
    )

    uploaded_file = st.file_uploader(
        "Upload equipment dataset",
        type=["csv"],
        help="Maximum upload size: 25 MB.",
        key="batch_csv_upload",
    )

    if uploaded_file is not None:

        st.info(
            f"Selected file: {uploaded_file.name} "
            f"({uploaded_file.size:,} bytes)"
        )

        if uploaded_file.size > 25 * 1024 * 1024:
            st.error("The file exceeds the 25 MB upload limit.")

        elif st.button(
            "Run batch prediction",
            use_container_width=True,
        ):

            file_bytes = uploaded_file.getvalue()

            with st.spinner(
                "Processing equipment dataset. Please wait..."
            ):
                try:
                    result = api.predict_csv(
                        uploaded_file.name,
                        file_bytes,
                    )

                    st.session_state.batch_result = result
                    st.session_state.batch_filename = uploaded_file.name
                    st.session_state.batch_file_bytes = file_bytes

                    st.session_state.prediction_download = None
                    st.session_state.rejected_download = None

                    st.success("Batch prediction completed.")

                except APIClientError as exc:
                    st.error(str(exc))

    if st.session_state.batch_result:

        st.markdown("---")

        result = st.session_state.batch_result

        display_batch_summary(result)

        st.markdown("---")

        display_batch_tables(result)

        st.markdown("---")

        st.markdown("#### Download reports")

        filename = st.session_state.batch_filename
        file_bytes = st.session_state.batch_file_bytes

        st.caption(
            "The current download endpoints recompute predictions. "
            "Downloading may therefore require additional processing time."
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Prepare prediction CSV",
                use_container_width=True,
            ):

                with st.spinner("Generating prediction CSV..."):

                    try:
                        content = api.download_predictions(
                            filename,
                            file_bytes,
                        )

                        st.session_state.prediction_download = content

                    except APIClientError as exc:
                        st.error(str(exc))

            if st.session_state.prediction_download is not None:

                st.download_button(
                    "Download predictions",
                    data=st.session_state.prediction_download,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col2:

            if st.button(
                "Prepare rejected rows CSV",
                use_container_width=True,
            ):

                with st.spinner("Generating rejected rows CSV..."):

                    try:
                        content = api.download_rejected_rows(
                            filename,
                            file_bytes,
                        )

                        st.session_state.rejected_download = content

                    except APIClientError as exc:
                        st.error(str(exc))

            if st.session_state.rejected_download is not None:

                st.download_button(
                    "Download rejected rows",
                    data=st.session_state.rejected_download,
                    file_name="rejected_rows.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Predictive Maintenance | FastAPI + Registered ML Models "
    "| Decision-support dashboard, not a certified safety-control system."
)
