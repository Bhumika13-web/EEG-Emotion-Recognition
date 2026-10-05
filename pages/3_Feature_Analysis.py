import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Feature Analysis",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# CONSTANTS
# ============================================================

CHANNELS = [
    "FP1", "AF3", "F3", "F7",
    "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "OZ", "O2",
    "PO4", "P8", "P4", "CP6",
    "CP2", "C4", "T8", "FC6",
    "FC2", "F4", "F8", "AF4",
    "FP2", "FZ", "CZ", "PZ",
]


BANDS = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
]


FEATURE_NAMES = [
    "DE - Delta",
    "DE - Theta",
    "DE - Alpha",
    "DE - Beta",
    "DE - Gamma",
    "PSD - Delta",
    "PSD - Theta",
    "PSD - Alpha",
    "PSD - Beta",
    "PSD - Gamma",
]


DE_INDICES = [0, 1, 2, 3, 4]

PSD_INDICES = [5, 6, 7, 8, 9]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.feature-header {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 25px;
}

.feature-header-title {
    font-size: 32px;
    font-weight: 800;
}

.feature-header-text {
    color: #cbd5e1;
    margin-top: 8px;
}

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 20px;
    min-height: 120px;
    box-shadow: 0 3px 12px rgba(15,23,42,0.05);
}

.info-title {
    color: #64748b;
    font-size: 13px;
}

.info-value {
    color: #0f172a;
    font-size: 26px;
    font-weight: 800;
    margin-top: 5px;
}

.section-title {
    font-size: 24px;
    font-weight: 750;
    color: #0f172a;
    margin-top: 30px;
    margin-bottom: 12px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="feature-header">

<div class="feature-header-title">
📊 EEG Feature Analysis
</div>

<div class="feature-header-text">
Analysis of Differential Entropy (DE) and
Power Spectral Density (PSD) features across
EEG frequency bands and channels.
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DEAP
# ============================================================

@st.cache_data
def load_deap():

    file_path = PROCESSED_DIR / "deap_train.npz"

    if not file_path.exists():
        return None

    data = np.load(file_path)

    return {
        "features": data["features"],
        "valence": data["valence"],
        "arousal": data["arousal"],
        "subjects": data["subject_ids"],
        "trials": data["trial_ids"],
    }


data = load_deap()


if data is None:

    st.error(
        "DEAP processed dataset was not found."
    )

    st.stop()


features = data["features"]
valence = data["valence"]
arousal = data["arousal"]
subjects = data["subjects"]
trials = data["trials"]


# ============================================================
# DATA INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">Feature Dataset</div>',
    unsafe_allow_html=True,
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        """
<div class="info-card">
<div class="info-title">Samples</div>
<div class="info-value">960</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        """
<div class="info-card">
<div class="info-title">EEG Channels</div>
<div class="info-value">32</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        """
<div class="info-card">
<div class="info-title">Windows / Sample</div>
<div class="info-value">30</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        """
<div class="info-card">
<div class="info-title">Features / Channel</div>
<div class="info-value">10</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.info(
    "Representation: 30 temporal windows × 32 EEG channels × "
    "10 features (5 Differential Entropy + 5 PSD)."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Feature Controls")


selected_subject = st.sidebar.selectbox(
    "Select Subject",
    sorted(
        np.unique(subjects).tolist()
    ),
)


subject_indices = np.where(
    subjects == selected_subject
)[0]


selected_trial_number = st.sidebar.selectbox(
    "Select Trial",
    list(range(len(subject_indices))),
)


sample_index = subject_indices[
    selected_trial_number
]


selected_window = st.sidebar.slider(
    "Select Window",
    min_value=0,
    max_value=features.shape[1] - 1,
    value=0,
)


sample = features[
    sample_index
]


window_features = sample[
    selected_window
]


# ============================================================
# SELECTED SAMPLE
# ============================================================

st.markdown(
    '<div class="section-title">Selected Sample</div>',
    unsafe_allow_html=True,
)


sc1, sc2, sc3, sc4 = st.columns(4)


with sc1:

    st.metric(
        "Subject",
        f"S{int(selected_subject) + 1:02d}",
    )


with sc2:

    st.metric(
        "Trial",
        int(selected_trial_number) + 1,
    )


with sc3:

    st.metric(
        "Window",
        selected_window + 1,
    )


with sc4:

    st.metric(
        "Feature Vector",
        "32 × 10",
    )


# ============================================================
# DE VS PSD OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">DE vs PSD Overview</div>',
    unsafe_allow_html=True,
)


de_mean = sample[:, :, DE_INDICES].mean(
    axis=(0, 1)
)


psd_mean = sample[:, :, PSD_INDICES].mean(
    axis=(0, 1)
)


comparison_df = pd.DataFrame(
    {
        "Frequency Band": BANDS,
        "Differential Entropy": de_mean,
        "PSD": psd_mean,
    }
)


fig = go.Figure()


fig.add_trace(
    go.Bar(
        x=BANDS,
        y=de_mean,
        name="Differential Entropy",
    )
)


fig.add_trace(
    go.Bar(
        x=BANDS,
        y=psd_mean,
        name="PSD",
    )
)


fig.update_layout(
    title="Average DE and PSD Across Frequency Bands",
    xaxis_title="Frequency Band",
    yaxis_title="Average Feature Value",
    barmode="group",
    height=480,
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# DE ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Differential Entropy Analysis</div>',
    unsafe_allow_html=True,
)


de_values = sample[
    :,
    :,
    DE_INDICES,
]


de_band_means = de_values.mean(
    axis=(0, 1)
)


de_df = pd.DataFrame(
    {
        "Frequency Band": BANDS,
        "Mean DE": de_band_means,
    }
)


fig_de = go.Figure()


fig_de.add_trace(
    go.Bar(
        x=de_df["Frequency Band"],
        y=de_df["Mean DE"],
        name="Differential Entropy",
    )
)


fig_de.update_layout(
    title="Differential Entropy by Frequency Band",
    xaxis_title="Frequency Band",
    yaxis_title="Mean DE",
    height=420,
)


st.plotly_chart(
    fig_de,
    use_container_width=True,
)


# ============================================================
# PSD ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Power Spectral Density Analysis</div>',
    unsafe_allow_html=True,
)


psd_values = sample[
    :,
    :,
    PSD_INDICES,
]


psd_band_means = psd_values.mean(
    axis=(0, 1)
)


psd_df = pd.DataFrame(
    {
        "Frequency Band": BANDS,
        "Mean PSD": psd_band_means,
    }
)


fig_psd = go.Figure()


fig_psd.add_trace(
    go.Bar(
        x=psd_df["Frequency Band"],
        y=psd_df["Mean PSD"],
        name="PSD",
    )
)


fig_psd.update_layout(
    title="Power Spectral Density by Frequency Band",
    xaxis_title="Frequency Band",
    yaxis_title="Mean PSD",
    height=420,
)


st.plotly_chart(
    fig_psd,
    use_container_width=True,
)


# ============================================================
# CHANNEL × BAND HEATMAP
# ============================================================

st.markdown(
    '<div class="section-title">Channel × Frequency Band Analysis</div>',
    unsafe_allow_html=True,
)


feature_type = st.radio(
    "Select Feature Type",
    ["Differential Entropy", "PSD"],
    horizontal=True,
)


if feature_type == "Differential Entropy":

    selected_matrix = window_features[
        :, DE_INDICES
    ].T

else:

    selected_matrix = window_features[
        :, PSD_INDICES
    ].T


heatmap = go.Figure(
    data=go.Heatmap(
        z=selected_matrix,
        x=CHANNELS,
        y=BANDS,
        colorbar_title="Value",
    )
)


heatmap.update_layout(
    title=(
        f"{feature_type} — "
        f"Window {selected_window + 1}"
    ),
    xaxis_title="EEG Channel",
    yaxis_title="Frequency Band",
    height=520,
)


st.plotly_chart(
    heatmap,
    use_container_width=True,
)


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Channel Analysis</div>',
    unsafe_allow_html=True,
)


channel_feature_type = st.selectbox(
    "Channel Feature",
    FEATURE_NAMES,
)


feature_index = FEATURE_NAMES.index(
    channel_feature_type
)


channel_values = sample[
    :,
    :,
    feature_index
].mean(axis=0)


channel_df = pd.DataFrame(
    {
        "Channel": CHANNELS,
        "Mean Value": channel_values,
    }
)


channel_df_sorted = channel_df.sort_values(
    "Mean Value",
    ascending=False,
)


fig_channel = go.Figure()


fig_channel.add_trace(
    go.Bar(
        x=channel_df_sorted["Channel"],
        y=channel_df_sorted["Mean Value"],
    )
)


fig_channel.update_layout(
    title=f"Average {channel_feature_type} by EEG Channel",
    xaxis_title="EEG Channel",
    yaxis_title="Mean Value",
    height=500,
)


st.plotly_chart(
    fig_channel,
    use_container_width=True,
)


# ============================================================
# EMOTION GROUP ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Feature vs Emotion</div>',
    unsafe_allow_html=True,
)


emotion_feature = st.selectbox(
    "Select Feature",
    FEATURE_NAMES,
)


emotion_feature_index = FEATURE_NAMES.index(
    emotion_feature
)


sample_values = features[
    :,
    :,
    :,
    emotion_feature_index
].mean(axis=(1, 2))


emotion_df = pd.DataFrame(
    {
        "Value": sample_values,
        "Valence": valence,
        "Arousal": arousal,
    }
)


emotion_df["Valence Group"] = np.where(
    emotion_df["Valence"] == 1,
    "High Valence",
    "Low Valence",
)


emotion_df["Arousal Group"] = np.where(
    emotion_df["Arousal"] == 1,
    "High Arousal",
    "Low Arousal",
)


emotion_view = st.radio(
    "Compare By",
    ["Valence", "Arousal"],
    horizontal=True,
)


if emotion_view == "Valence":

    groups = [
        "Low Valence",
        "High Valence",
    ]

    values = [
        emotion_df.loc[
            emotion_df["Valence Group"] == group,
            "Value",
        ].mean()
        for group in groups
    ]

else:

    groups = [
        "Low Arousal",
        "High Arousal",
    ]

    values = [
        emotion_df.loc[
            emotion_df["Arousal Group"] == group,
            "Value",
        ].mean()
        for group in groups
    ]


fig_emotion = go.Figure()


fig_emotion.add_trace(
    go.Bar(
        x=groups,
        y=values,
        text=[
            f"{value:.4f}"
            for value in values
        ],
        textposition="aut✓,
    )
)


fig_emotion.update_layout(
    title=(
        f"{emotion_feature} vs "
        f"{emotion_view}"
    ),
    xaxis_title=emotion_view,
    yaxis_title="Average Feature Value",
    height=430,
)


st.plotly_chart(
    fig_emotion,
    use_container_width=True,
)


# ============================================================
# FEATURE STATISTICS TABLE
# ============================================================

st.markdown(
    '<div class="section-title">Feature Statistics</div>',
    unsafe_allow_html=True,
)


stats_rows = []


for i, name in enumerate(FEATURE_NAMES):

    values = features[
        :, :, :, i
    ].flatten()

    stats_rows.append(
        {
            "Feature": name,
            "Mean": np.mean(values),
            "Std": np.std(values),
            "Minimum": np.min(values),
            "Maximum": np.max(values),
            "Median": np.median(values),
        }
    )


stats_df = pd.DataFrame(
    stats_rows
)


display_stats = stats_df.copy()


for column in [
    "Mean",
    "Std",
    "Minimum",
    "Maximum",
    "Median",
]:

    display_stats[column] = display_stats[
        column
    ].round(4)


st.dataframe(
    display_stats,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Feature Analysis • DEAP • Differential Entropy + PSD"
)