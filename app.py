import streamlit as st
from pathlib import Path
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SESSION STATE
# ============================================================

if "dataset_uploaded" not in st.session_state:
    st.session_state.dataset_uploaded = False

if "dataset_processed" not in st.session_state:
    st.session_state.dataset_processed = False

if "dataset_type" not in st.session_state:
    st.session_state.dataset_type = None

if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None

if "processed_features" not in st.session_state:
    st.session_state.processed_features = None

if "processed_labels" not in st.session_state:
    st.session_state.processed_labels = None

if "dataset_info" not in st.session_state:
    st.session_state.dataset_info = {}

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   GENERAL
   ============================================================ */

.main {
    background-color: var(--st-background-color);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background-color: var(--st-secondary-background-color);
}

[data-testid="stSidebar"] * {
    color: var(--st-text-color);
}

.sidebar-brand {
    font-size: 25px;
    font-weight: 800;
    color: var(--st-heading-color) !important;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    font-size: 13px;
    color: var(--st-text-color) !important;
    opacity: 0.65;
    margin-bottom: 18px;
}

.sidebar-section {
    font-size: 15px;
    font-weight: 700;
    color: var(--st-heading-color) !important;
    margin-top: 20px;
    margin-bottom: 10px;
}

[data-testid="stSidebar"] [data-testid="stPageLink"] {
    border-radius: 10px;
    margin: 4px 0;
    padding: 2px 5px;
}

[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
    background-color: rgba(59, 130, 246, 0.12);
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554
    );
    border: 1px solid #334155;
    margin-bottom: 30px;
}

.hero-title {
    color: #ffffff !important;
    font-size: 40px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero-subtitle {
    color: #dbeafe !important;
    font-size: 19px;
    margin-bottom: 10px;
    line-height: 1.5;
}

.hero-tech {
    color: #93c5fd !important;
    font-size: 15px;
}

/* ============================================================
   SECTION TITLES
   ============================================================ */

.section-title {
    font-size: 26px;
    font-weight: 750;
    color: var(--st-heading-color) !important;
    margin-top: 30px;
    margin-bottom: 15px;
}

/* ============================================================
   METRIC CARDS
   ============================================================ */

.metric-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
    min-height: 120px;
}

.metric-title {
    color: var(--st-text-color) !important;
    opacity: 0.72;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--st-heading-color) !important;
    font-size: 30px;
    font-weight: 800;
}

.metric-description {
    color: var(--st-text-color) !important;
    opacity: 0.72;
    font-size: 13px;
    margin-top: 6px;
}

/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 15px;
    padding: 20px 10px;
    text-align: center;
    min-height: 145px;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
}

.pipeline-number {
    color: #3b82f6 !important;
    font-size: 26px;
    font-weight: 800;
}

.pipeline-title {
    color: var(--st-heading-color) !important;
    font-size: 16px;
    font-weight: 700;
    margin-top: 8px;
}

.pipeline-description {
    color: var(--st-text-color) !important;
    opacity: 0.72;
    font-size: 13px;
    margin-top: 5px;
}

/* ============================================================
   DATASET CARDS
   ============================================================ */

.dataset-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 25px;
    min-height: 230px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.08);
}

.dataset-title {
    font-size: 23px;
    font-weight: 750;
    color: var(--st-heading-color) !important;
    margin-bottom: 15px;
}

.dataset-text {
    color: var(--st-text-color) !important;
    opacity: 0.85;
    line-height: 1.9;
    font-size: 15px;
}

/* ============================================================
   BEST MODEL
   ============================================================ */

.best-model {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 18px;
    padding: 25px;
    margin-top: 10px;
}

/* ============================================================
   PROJECT STATUS
   ============================================================ */

.status-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    min-height: 90px;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid var(--st-border-color);
    color: var(--st-text-color) !important;
    opacity: 0.70;
    font-size: 13px;
}

/* ============================================================
   STREAMLIT COMPONENTS
   ============================================================ */

[data-testid="stMetricLabel"] {
    color: var(--st-text-color) !important;
    opacity: 0.75;
}

[data-testid="stMetricValue"] {
    color: var(--st-heading-color) !important;
}

[data-testid="stDataFrame"] {
    border-color: var(--st-border-color);
}

hr {
    border-color: var(--st-border-color) !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">
            EEG Lab
        </div>

        <div class="sidebar-subtitle">
            EEG Emotion Recognition
        </div>
        """
    )

    st.divider()

    st.html(
        """
        <div class="sidebar-section">
            Navigation
        </div>
        """
    )

    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    st.page_link(
        "app.py",
        label="Dashboard",
        icon="",
    )

    # --------------------------------------------------------
    # UPLOAD DATASET
    # --------------------------------------------------------

    st.page_link(
        "pages/1_Upload_Dataset.py",
        label="Upload Dataset",
        icon="",
    )

    # --------------------------------------------------------
    # EEG VISUALIZATION
    # --------------------------------------------------------

    st.page_link(
        "pages/2_EEG_Visualization.py",
        label="EEG Visualization",
        icon="",
    )

    # --------------------------------------------------------
    # FEATURE ANALYSIS
    # --------------------------------------------------------

    st.page_link(
        "pages/3_Feature_Analysis.py",
        label="Feature Analysis",
        icon="",
    )

    # --------------------------------------------------------
    # EMOTION PREDICTION
    # --------------------------------------------------------

    st.page_link(
        "pages/4_Emotion_Prediction.py",
        label="Emotion Prediction",
        icon="",
    )

    # --------------------------------------------------------
    # MODEL ARCHITECTURE
    # --------------------------------------------------------

    st.page_link(
        "pages/5_Model_Architecture.py",
        label="Model Architecture",
        icon="",
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    st.page_link(
        "pages/6_Evaluation.py",
        label="Evaluation",
        icon="",
    )

    st.divider()

    st.caption(
        "EEG based Emotion Recognition\n"
        "Spatial-Temporal Representation Learning"
    )

# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-title">
            EEG Emotion Recognition
        </div>

        <div class="hero-subtitle">
            Spatial-Temporal Representation Learning
            for EEG-based Emotion Classification
        </div>

        <div class="hero-tech">
            DEAP &nbsp;&nbsp;
            SEED &nbsp;&nbsp;
            Differential Entropy &nbsp;&nbsp;
            PSD &nbsp;&nbsp;
            GCN &nbsp;&nbsp;
            GRU
        </div>
    </div>
    """
)

# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.html(
    """
    <div class="section-title">
        Project Overview
    </div>
    """
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-title">
                Datasets
            </div>

            <div class="metric-value">
                2
            </div>

            <div class="metric-description">
                DEAP + SEED
            </div>

        </div>
        """
    )

with col2:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-title">
                DEAP Subjects
            </div>

            <div class="metric-value">
                32
            </div>

            <div class="metric-description">
                EEG recordings
            </div>

        </div>
        """
    )

with col3:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-title">
                SEED Subjects
            </div>

            <div class="metric-value">
                15
            </div>

            <div class="metric-description">
                Unseen-subject evaluation
            </div>

        </div>
        """
    )

with col4:

    st.html(
        """
        <div class="metric-card">

            <div class="metric-title">
                Best SEED Model
            </div>

            <div class="metric-value">
                GCN
            </div>

            <div class="metric-description">
                44.74% Accuracy
            </div>

        </div>
        """
    )

# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.html(
    """
    <div class="section-title">
        System Pipeline
    </div>
    """
)

pipeline = [
    ("01", "Dataset", "Upload DEAP / SEED"),
    ("02", "Preprocessing", "Signal preparation"),
    ("03", "Features", "DE + PSD"),
    ("04", "Spatial Learning", "Electrode GCN"),
    ("05", "Temporal Learning", "GRU"),
    ("06", "Classification", "Emotion prediction"),
]

cols = st.columns(6)

for col, (number, title, description) in zip(
    cols,
    pipeline,
):

    with col:

        st.html(
            f"""
            <div class="pipeline-card">

                <div class="pipeline-number">
                    {number}
                </div>

                <div class="pipeline-title">
                    {title}
                </div>

                <div class="pipeline-description">
                    {description}
                </div>

            </div>
            """
        )

# ============================================================
# DATASETS
# ============================================================

st.html(
    """
    <div class="section-title">
        Datasets
    </div>
    """
)

dataset_col1, dataset_col2 = st.columns(2)

with dataset_col1:

    st.html(
        """
        <div class="dataset-card">

            <div class="dataset-title">
                DEAP
            </div>

            <div class="dataset-text">

                <b>32</b> subjects<br>
                <b>32</b> EEG channels<br>
                <b>40</b> trials per subject<br>
                <b>128 Hz</b> sampling rate<br>
                Valence and Arousal classification<br>
                Differential Entropy + PSD features

            </div>

        </div>
        """
    )

with dataset_col2:

    st.html(
        """
        <div class="dataset-card">

            <div class="dataset-title">
                SEED
            </div>

            <div class="dataset-text">

                <b>15</b> subjects<br>
                <b>62</b> EEG channels<br>
                <b>3</b> emotion classes<br>
                Negative / Neutral / Positive<br>
                Differential Entropy representation<br>
                Spatial electrode graph

            </div>

        </div>
        """
    )

# ============================================================
# CURRENT UPLOAD STATUS
# ============================================================

st.html(
    """
    <div class="section-title">
        Current Session
    </div>
    """
)

if st.session_state.dataset_processed:

    dataset_name = st.session_state.dataset_type

    features = st.session_state.processed_features

    labels = st.session_state.processed_labels

    info = st.session_state.dataset_info

    st.success(
        f"{dataset_name} dataset processed successfully."
    )

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    if dataset_name == "DEAP":

        with status_col1:

            st.metric(
                "Dataset",
                "DEAP",
            )

        with status_col2:

            st.metric(
                "Trials",
                info.get("trials", "-"),
            )

        with status_col3:

            st.metric(
                "Channels",
                info.get("channels", "-"),
            )

        with status_col4:

            st.metric(
                "Features",
                info.get("features_per_window", "-"),
            )

    else:

        with status_col1:

            st.metric(
                "Dataset",
                "SEED",
            )

        with status_col2:

            st.metric(
                "Samples",
                f"{info.get('samples', 0):,}",
            )

        with status_col3:

            st.metric(
                "Channels",
                info.get("channels", "-"),
            )

        with status_col4:

            st.metric(
                "Classes",
                info.get("classes", "-"),
            )

    st.info(
        "Your processed dataset is available in the "
        "current Streamlit session. Open the other pages "
        "from the sidebar to continue analysis."
    )

else:

    st.info(
        "No dataset has been uploaded in the current session. "
        "Open 'Upload Dataset' from the sidebar to begin."
    )

# ============================================================
# BEST SEED MODEL
# ============================================================

st.html(
    """
    <div class="section-title">
        Best SEED Model
    </div>
    """
)

seed_file = (
    Path(__file__).resolve().parent
    / "results"
    / "SEED"
    / "seed_model_comparison.csv"
)

if seed_file.exists():

    try:

        seed_df = pd.read_csv(
            seed_file
        )

        if (
            "Macro F1" in seed_df.columns
            and
            not seed_df.empty
        ):

            best_idx = seed_df[
                "Macro F1"
            ].idxmax()

            best_model = seed_df.loc[
                best_idx
            ]

            st.html(
                """
                <div class="best-model">
                """
            )

            model_col1, model_col2, model_col3, model_col4 = (
                st.columns(4)
            )

            with model_col1:

                st.metric(
                    "Model",
                    str(
                        best_model["Model"]
                    ),
                )

            with model_col2:

                st.metric(
                    "Accuracy",
                    f"{best_model['Accuracy'] * 100:.2f}%",
                )

            with model_col3:

                st.metric(
                    "Balanced Accuracy",
                    f"{best_model['Balanced Accuracy'] * 100:.2f}%",
                )

            with model_col4:

                st.metric(
                    "Macro F1",
                    f"{best_model['Macro F1'] * 100:.2f}%",
                )

            st.html(
                """
                </div>
                """
            )

        else:

            st.warning(
                "SEED comparison file does not contain "
                "the expected evaluation columns."
            )

    except Exception as error:

        st.warning(
            f"Unable to read SEED results: {error}"
        )

else:

    st.warning(
        "SEED evaluation results were not found."
    )

# ============================================================
# PROJECT STATUS
# ============================================================

st.html(
    """
    <div class="section-title">
        Project Status
    </div>
    """
)

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:

    st.success(
        "Dataset Processing"
    )

with status_col2:

    st.success(
        "Model Evaluation"
    )

with status_col3:

    st.success(
        "Interactive Dashboard"
    )

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        EEG based Emotion Recognition using
        Spatial-Temporal Representation Learning

        <br><br>

        Academic Project &nbsp; DEAP + SEED &nbsp; GCN / GRU

    </div>
    """
)