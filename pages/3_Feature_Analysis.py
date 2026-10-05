import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Feature Analysis",
    page_icon="",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DEAP_CHANNELS = [
    "FP1", "AF3", "F3", "F7",
    "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7",
    "PO3", "O1", "OZ", "O2",
    "PO4", "P8", "P4", "CP6",
    "CP2", "C4", "T8", "FC6",
    "FC2", "F4", "F8", "AF4",
    "FP2", "FZ", "CZ", "PZ",
]

SEED_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "O1", "OZ", "O2",
    "CB1", "CB2",
]

BANDS = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
]

DEAP_FEATURE_NAMES = [
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

DEAP_DE_INDICES = [0, 1, 2, 3, 4]
DEAP_PSD_INDICES = [5, 6, 7, 8, 9]

SEED_EMOTION_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


# ============================================================
# HEADER
# ============================================================

st.title("EEG Feature Analysis")

st.write(
    "Analysis of EEG feature representations across "
    "frequency bands and EEG channels."
)

st.divider()


# ============================================================
# GET DATA FROM SESSION STATE
# ============================================================

features = st.session_state.get(
    "processed_features",
    None,
)

labels = st.session_state.get(
    "processed_labels",
    None,
)

subject_ids = st.session_state.get(
    "processed_subject_ids",
    None,
)

dataset_type = st.session_state.get(
    "dataset_type",
    None,
)

dataset_processed = st.session_state.get(
    "dataset_processed",
    False,
)

processed_source = st.session_state.get(
    "processed_source",
    None,
)


# ============================================================
# CHECK DATA
# ============================================================

if (
    not dataset_processed
    or features is None
    or dataset_type is None
):

    st.warning(
        "No processed EEG dataset is currently loaded."
    )

    st.info(
        "Please open **Upload Dataset**, select DEAP or SEED, "
        "upload the required dataset, process it, and then "
        "return to Feature Analysis."
    )

    st.stop()


# ============================================================
# CONVERT TO NUMPY
# ============================================================

features = np.asarray(features)

if labels is not None:
    labels = np.asarray(labels)

if subject_ids is not None:
    subject_ids = np.asarray(subject_ids)

dataset_type = str(
    dataset_type
).strip().upper()


# ============================================================
# CURRENT DATASET
# ============================================================

st.header("Current Dataset")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "Dataset",
        dataset_type,
    )

with info2:
    st.metric(
        "Samples / Trials",
        f"{len(features):,}",
    )

with info3:
    st.metric(
        "Representation",
        str(features.shape[1:]),
    )

with info4:

    if dataset_type == "DEAP":

        st.metric(
            "Channels",
            32,
        )

    elif dataset_type == "SEED":

        st.metric(
            "Channels",
            62,
        )

if processed_source:

    st.caption(
        f"Data source: {processed_source}"
    )


# ============================================================
# ============================================================
# DEAP FEATURE ANALYSIS
# ============================================================
# ============================================================

if dataset_type == "DEAP":

    # --------------------------------------------------------
    # NORMALIZE DEAP SHAPE
    # --------------------------------------------------------

    if features.ndim == 4:

        if (
            features.shape[2] == 32
            and features.shape[3] == 10
        ):

            deap_features = features

        else:

            st.error(
                "Unexpected DEAP feature shape. "
                f"Received {features.shape}. "
                "Expected (trials, windows, 32, 10)."
            )

            st.stop()

    elif features.ndim == 3:

        if features.shape[-1] == 320:

            deap_features = features.reshape(
                features.shape[0],
                features.shape[1],
                32,
                10,
            )

        else:

            st.error(
                "Unexpected DEAP feature shape. "
                f"Received {features.shape}. "
                "Expected (..., 320) or (..., 32, 10)."
            )

            st.stop()

    else:

        st.error(
            "Unexpected DEAP feature dimensions: "
            f"{features.shape}"
        )

        st.stop()


    # --------------------------------------------------------
    # DEAP DATASET INFORMATION
    # --------------------------------------------------------

    trials = deap_features.shape[0]
    windows = deap_features.shape[1]

    st.header("DEAP Feature Dataset")

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Trials",
            trials,
        )

    with d2:

        st.metric(
            "EEG Channels",
            32,
        )

    with d3:

        st.metric(
            "Windows / Trial",
            windows,
        )

    with d4:

        st.metric(
            "Features / Channel",
            10,
        )

    st.info(
        "DEAP representation: 30 temporal windows × "
        "32 EEG channels × 10 features "
        "(5 Differential Entropy + 5 PSD)."
    )


    # --------------------------------------------------------
    # FEATURE CONTROLS
    # --------------------------------------------------------

    st.header("Feature Controls")

    control1, control2 = st.columns(2)

    with control1:

        selected_trial = st.selectbox(
            "Select Trial",
            range(1, trials + 1),
            key="feature_deap_trial",
        )

    with control2:

        selected_window = st.selectbox(
            "Select EEG Window",
            range(1, windows + 1),
            key="feature_deap_window",
        )

    trial_index = selected_trial - 1
    window_index = selected_window - 1

    selected_trial_data = deap_features[
        trial_index
    ]

    selected_window_data = deap_features[
        trial_index,
        window_index,
    ]


    # --------------------------------------------------------
    # SELECTED SAMPLE
    # --------------------------------------------------------

    st.header("Selected Sample")

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Trial",
            selected_trial,
        )

    with s2:

        st.metric(
            "Window",
            selected_window,
        )

    with s3:

        st.metric(
            "Channels",
            32,
        )

    with s4:

        st.metric(
            "Features",
            10,
        )


    # --------------------------------------------------------
    # DE VS PSD
    # --------------------------------------------------------

    st.header("DE vs PSD Overview")

    de_mean = selected_trial_data[
        :,
        :,
        DEAP_DE_INDICES,
    ].mean(
        axis=(0, 1)
    )

    psd_mean = selected_trial_data[
        :,
        :,
        DEAP_PSD_INDICES,
    ].mean(
        axis=(0, 1)
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


    # --------------------------------------------------------
    # DIFFERENTIAL ENTROPY
    # --------------------------------------------------------

    st.header("Differential Entropy Analysis")

    de_values = selected_trial_data[
        :,
        :,
        DEAP_DE_INDICES,
    ]

    de_band_means = de_values.mean(
        axis=(0, 1)
    )

    fig_de = go.Figure()

    fig_de.add_trace(
        go.Bar(
            x=BANDS,
            y=de_band_means,
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


    # --------------------------------------------------------
    # PSD
    # --------------------------------------------------------

    st.header("Power Spectral Density Analysis")

    psd_values = selected_trial_data[
        :,
        :,
        DEAP_PSD_INDICES,
    ]

    psd_band_means = psd_values.mean(
        axis=(0, 1)
    )

    fig_psd = go.Figure()

    fig_psd.add_trace(
        go.Bar(
            x=BANDS,
            y=psd_band_means,
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


    # --------------------------------------------------------
    # CHANNEL × FREQUENCY HEATMAP
    # --------------------------------------------------------

    st.header("Channel × Frequency Band Heatmap")

    heatmap_type = st.selectbox(
        "Select Feature Type",
        [
            "Differential Entropy",
            "PSD",
        ],
        key="deap_heatmap_type",
    )

    if heatmap_type == "Differential Entropy":

        matrix = selected_window_data[
            :,
            DEAP_DE_INDICES,
        ].T

    else:

        matrix = selected_window_data[
            :,
            DEAP_PSD_INDICES,
        ].T

    heatmap = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=DEAP_CHANNELS,
            y=BANDS,
            colorbar_title="Value",
        )
    )

    heatmap.update_layout(
        title=(
            f"{heatmap_type} — "
            f"Window {selected_window}"
        ),
        xaxis_title="EEG Channel",
        yaxis_title="Frequency Band",
        height=520,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # CHANNEL ANALYSIS
    # --------------------------------------------------------

    st.header("Channel Analysis")

    channel_feature = st.selectbox(
        "Select Feature",
        DEAP_FEATURE_NAMES,
        key="deap_channel_feature",
    )

    feature_index = DEAP_FEATURE_NAMES.index(
        channel_feature
    )

    channel_values = selected_trial_data[
        :,
        :,
        feature_index,
    ].mean(
        axis=0
    )

    channel_df = pd.DataFrame(
        {
            "Channel": DEAP_CHANNELS,
            "Mean Value": channel_values,
        }
    )

    channel_df = channel_df.sort_values(
        "Mean Value",
        ascending=False,
    )

    fig_channel = go.Figure()

    fig_channel.add_trace(
        go.Bar(
            x=channel_df["Channel"],
            y=channel_df["Mean Value"],
        )
    )

    fig_channel.update_layout(
        title=f"Average {channel_feature} by EEG Channel",
        xaxis_title="EEG Channel",
        yaxis_title="Mean Value",
        height=500,
    )

    st.plotly_chart(
        fig_channel,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # FEATURE VS EMOTION
    # --------------------------------------------------------

    if labels is not None:

        labels_array = np.asarray(
            labels
        )

        if (
            labels_array.ndim >= 2
            and len(labels_array) == trials
            and labels_array.shape[1] >= 2
        ):

            st.header("Feature vs Emotion")

            emotion_feature = st.selectbox(
                "Select Feature",
                DEAP_FEATURE_NAMES,
                key="deap_emotion_feature",
            )

            emotion_feature_index = (
                DEAP_FEATURE_NAMES.index(
                    emotion_feature
                )
            )

            feature_values = deap_features[
                :,
                :,
                :,
                emotion_feature_index,
            ].mean(
                axis=(1, 2)
            )

            raw_valence = labels_array[:, 0]
            raw_arousal = labels_array[:, 1]

            # ------------------------------------------------
            # VALENCE GROUP
            # ------------------------------------------------

            if np.all(
                np.isin(
                    np.unique(raw_valence),
                    [0, 1],
                )
            ):

                valence_group = np.where(
                    raw_valence == 1,
                    "High Valence",
                    "Low Valence",
                )

            else:

                valence_group = np.where(
                    raw_valence >= 5.0,
                    "High Valence",
                    "Low Valence",
                )


            # ------------------------------------------------
            # AROUSAL GROUP
            # ------------------------------------------------

            if np.all(
                np.isin(
                    np.unique(raw_arousal),
                    [0, 1],
                )
            ):

                arousal_group = np.where(
                    raw_arousal == 1,
                    "High Arousal",
                    "Low Arousal",
                )

            else:

                arousal_group = np.where(
                    raw_arousal >= 5.0,
                    "High Arousal",
                    "Low Arousal",
                )


            emotion_view = st.radio(
                "Compare By",
                [
                    "Valence",
                    "Arousal",
                ],
                horizontal=True,
                key="deap_emotion_view",
            )


            if emotion_view == "Valence":

                groups = [
                    "Low Valence",
                    "High Valence",
                ]

                group_values = []

                for group in groups:

                    mask = (
                        valence_group == group
                    )

                    if np.any(mask):

                        group_values.append(
                            float(
                                np.mean(
                                    feature_values[
                                        mask
                                    ]
                                )
                            )
                        )

                    else:

                        group_values.append(
                            np.nan
                        )

            else:

                groups = [
                    "Low Arousal",
                    "High Arousal",
                ]

                group_values = []

                for group in groups:

                    mask = (
                        arousal_group == group
                    )

                    if np.any(mask):

                        group_values.append(
                            float(
                                np.mean(
                                    feature_values[
                                        mask
                                    ]
                                )
                            )
                        )

                    else:

                        group_values.append(
                            np.nan
                        )


            fig_emotion = go.Figure()

            fig_emotion.add_trace(
                go.Bar(
                    x=groups,
                    y=group_values,
                    text=[
                        (
                            f"{value:.4f}"
                            if np.isfinite(value)
                            else "N/A"
                        )
                        for value in group_values
                    ],
                    textposition="auto",
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


    # --------------------------------------------------------
    # FEATURE STATISTICS
    # --------------------------------------------------------

    st.header("Feature Statistics")

    statistics = []

    for index, feature_name in enumerate(
        DEAP_FEATURE_NAMES
    ):

        values = deap_features[
            :,
            :,
            :,
            index,
        ]

        statistics.append(
            {
                "Feature": feature_name,
                "Mean": float(
                    np.mean(values)
                ),
                "Std": float(
                    np.std(values)
                ),
                "Minimum": float(
                    np.min(values)
                ),
                "Maximum": float(
                    np.max(values)
                ),
            }
        )

    stats_df = pd.DataFrame(
        statistics
    )

    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# ============================================================
# SEED FEATURE ANALYSIS
# ============================================================
# ============================================================

elif dataset_type == "SEED":

    # --------------------------------------------------------
    # VALIDATE SEED
    # --------------------------------------------------------

    if features.ndim != 3:

        st.error(
            "Unexpected SEED feature shape. "
            f"Received {features.shape}. "
            "Expected (samples, 5, 62)."
        )

        st.stop()


    if features.shape[1:] != (
        5,
        62,
    ):

        st.error(
            "Unexpected SEED representation. "
            f"Received {features.shape}. "
            "Expected (samples, 5, 62)."
        )

        st.stop()


    samples = features.shape[0]


    # --------------------------------------------------------
    # SEED DATASET INFORMATION
    # --------------------------------------------------------

    st.header("SEED Feature Dataset")

    s1, s2, s3, s4 = st.columns(4)

    with s1:

        st.metric(
            "Samples",
            f"{samples:,}",
        )

    with s2:

        st.metric(
            "Frequency Bands",
            5,
        )

    with s3:

        st.metric(
            "EEG Channels",
            62,
        )

    with s4:

        st.metric(
            "Emotion Classes",
            3,
        )

    st.info(
        "SEED representation: 5 frequency-band features "
        "× 62 EEG channels. The uploaded SEED representation "
        "is already provided as Differential Entropy features."
    )


    # --------------------------------------------------------
    # FEATURE CONTROLS
    # --------------------------------------------------------

    st.header("Feature Controls")

    selected_sample_number = st.selectbox(
        "Select SEED Sample",
        range(
            1,
            samples + 1,
        ),
        key="feature_seed_sample",
    )

    sample_index = (
        selected_sample_number - 1
    )

    selected_sample = features[
        sample_index
    ]


    # --------------------------------------------------------
    # SUBJECT INFORMATION
    # --------------------------------------------------------

    subject_value = None

    if (
        subject_ids is not None
        and len(subject_ids) == samples
    ):

        subject_value = int(
            subject_ids[
                sample_index
            ]
        )


    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Sample",
            selected_sample_number,
        )

    with c2:

        if subject_value is not None:

            st.metric(
                "Subject",
                subject_value,
            )

        else:

            st.metric(
                "Subject",
                "Available in upload",
            )

    with c3:

        st.metric(
            "Input Shape",
            "5 × 62",
        )


    # --------------------------------------------------------
    # REFERENCE EMOTION
    # --------------------------------------------------------

    if (
        labels is not None
        and len(labels) == samples
    ):

        label_value = int(
            np.asarray(
                labels
            ).reshape(-1)[
                sample_index
            ]
        )

        emotion = SEED_EMOTION_MAP.get(
            label_value,
            f"Class {label_value}",
        )

        st.header("Reference Emotion")

        e1, e2 = st.columns(2)

        with e1:

            st.metric(
                "Emotion",
                emotion,
            )

        with e2:

            st.metric(
                "Class",
                label_value,
            )


    # --------------------------------------------------------
    # FREQUENCY BAND ANALYSIS
    # --------------------------------------------------------

    st.header("Frequency Band Analysis")

    band_values = selected_sample.mean(
        axis=1
    )

    band_df = pd.DataFrame(
        {
            "Frequency Band": BANDS,
            "Mean DE": band_values,
        }
    )

    fig_band = go.Figure()

    fig_band.add_trace(
        go.Bar(
            x=BANDS,
            y=band_values,
            name="Differential Entropy",
        )
    )

    fig_band.update_layout(
        title="Average Differential Entropy by Frequency Band",
        xaxis_title="Frequency Band",
        yaxis_title="Mean DE",
        height=430,
    )

    st.plotly_chart(
        fig_band,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # BAND × CHANNEL HEATMAP
    # --------------------------------------------------------

    st.header("Frequency Band × EEG Channel")

    heatmap = go.Figure(
        data=go.Heatmap(
            z=selected_sample,
            x=SEED_CHANNELS,
            y=BANDS,
            colorbar_title="DE",
        )
    )

    heatmap.update_layout(
        title="SEED Differential Entropy Heatmap",
        xaxis_title="EEG Channel",
        yaxis_title="Frequency Band",
        height=540,
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # CHANNEL ANALYSIS
    # --------------------------------------------------------

    st.header("Channel Analysis")

    channel_values = selected_sample.mean(
        axis=0
    )

    channel_df = pd.DataFrame(
        {
            "Channel": SEED_CHANNELS,
            "Mean DE": channel_values,
        }
    )

    channel_df = channel_df.sort_values(
        "Mean DE",
        ascending=False,
    )

    fig_channel = go.Figure()

    fig_channel.add_trace(
        go.Bar(
            x=channel_df["Channel"],
            y=channel_df["Mean DE"],
        )
    )

    fig_channel.update_layout(
        title="Average Differential Entropy by EEG Channel",
        xaxis_title="EEG Channel",
        yaxis_title="Mean DE",
        height=500,
    )

    st.plotly_chart(
        fig_channel,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # FEATURE STATISTICS
    # --------------------------------------------------------

    st.header("Feature Statistics")

    seed_statistics = []

    for band_index, band_name in enumerate(
        BANDS
    ):

        values = features[
            :,
            band_index,
            :,
        ]

        seed_statistics.append(
            {
                "Frequency Band": band_name,
                "Mean": float(
                    np.mean(values)
                ),
                "Std": float(
                    np.std(values)
                ),
                "Minimum": float(
                    np.min(values)
                ),
                "Maximum": float(
                    np.max(values)
                ),
            }
        )

    seed_stats_df = pd.DataFrame(
        seed_statistics
    )

    st.dataframe(
        seed_stats_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# INVALID DATASET
# ============================================================

else:

    st.error(
        f"Unsupported dataset type: {dataset_type}"
    )

    st.stop()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EEG based Emotion Recognition using "
    "Spatial-Temporal Representation Learning"
)