import streamlit as st

import numpy as np

import pandas as pd

import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EEG Visualization",
    page_icon="",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ============================================================
   PAGE HEADER
   ============================================================ */

.page-title {
    font-size: 36px;
    font-weight: 800;
    color: var(--st-heading-color) !important;
    margin-bottom: 8px;
}

.page-subtitle {
    font-size: 16px;
    color: var(--st-text-color) !important;
    opacity: 0.72;
    margin-bottom: 28px;
}


/* ============================================================
   SECTION TITLE
   ============================================================ */

.section-title {
    font-size: 24px;
    font-weight: 750;
    color: var(--st-heading-color) !important;
    margin-top: 30px;
    margin-bottom: 15px;
}


/* ============================================================
   SUCCESS CARD
   ============================================================ */

.success-card {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.30);
    border-radius: 15px;
    padding: 18px 22px;
    margin: 15px 0 25px 0;
}

.success-title {
    color: #16a34a !important;
    font-size: 17px;
    font-weight: 750;
}

.success-text {
    color: var(--st-text-color) !important;
    opacity: 0.82;
    margin-top: 4px;
}


/* ============================================================
   LABEL CARDS
   ============================================================ */

.label-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 20px;
    min-height: 115px;
}

.label-name {
    color: var(--st-text-color) !important;
    opacity: 0.65;
    font-size: 13px;
    margin-bottom: 7px;
}

.label-value {
    color: var(--st-heading-color) !important;
    font-size: 28px;
    font-weight: 800;
}

.label-score {
    color: var(--st-text-color) !important;
    opacity: 0.65;
    font-size: 13px;
    margin-top: 5px;
}


/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 22px;
    margin-top: 10px;
}

.info-title {
    color: var(--st-heading-color) !important;
    font-size: 20px;
    font-weight: 750;
    margin-bottom: 12px;
}

.info-text {
    color: var(--st-text-color) !important;
    opacity: 0.82;
    line-height: 1.8;
    font-size: 15px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    color: var(--st-text-color) !important;
    opacity: 0.65;
    font-size: 13px;
    margin-top: 25px;
}


/* ============================================================
   STREAMLIT METRICS
   ============================================================ */

[data-testid="stMetricLabel"] {
    color: var(--st-text-color) !important;
}

[data-testid="stMetricValue"] {
    color: var(--st-heading-color) !important;
}


/* ============================================================
   DIVIDER
   ============================================================ */

hr {
    border-color: var(--st-border-color) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="page-title">EEG Visualization</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-subtitle">
        Visualize the processed EEG representation,
        frequency-band activity, electrode patterns,
        and emotion labels from the uploaded dataset.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GET PROCESSED DATA FROM UPLOAD PAGE
# ============================================================

features = st.session_state.get(
    "processed_features",
    None,
)

labels = st.session_state.get(
    "processed_labels",
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

subject_ids = st.session_state.get(
    "processed_subject_ids",
    None,
)


# ============================================================
# CHECK WHETHER DATASET EXISTS
# ============================================================

if (
    features is None
    or not dataset_processed
):

    st.warning(
        "No processed EEG dataset is currently loaded."
    )

    st.info(
        "Go to **Upload Dataset**, select DEAP or SEED, "
        "upload the required files, process the dataset, "
        "and then return to this page."
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


# ============================================================
# DATASET NAME
# ============================================================

dataset_name = str(
    dataset_type
).upper()


# ============================================================
# ============================================================
# DEAP VISUALIZATION
# ============================================================
# ============================================================

if dataset_name == "DEAP":

    # ========================================================
    # VALIDATE DEAP SHAPE
    # ========================================================

    if features.ndim == 3:

        trials = features.shape[0]

        windows = features.shape[1]

        flattened_features = features.shape[2]

        if flattened_features != 320:

            st.error(
                "The uploaded DEAP representation "
                "does not contain 320 values per window."
            )

            st.write(
                f"Received shape: {features.shape}"
            )

            st.write(
                "Expected: (trials, windows, 320)"
            )

            st.stop()

        channels = 32

        feature_count = 10

        # ----------------------------------------------------
        # Convert:
        #
        # (trials, windows, 320)
        #
        # to:
        #
        # (trials, windows, 32, 10)
        # ----------------------------------------------------

        features_4d = features.reshape(
            trials,
            windows,
            channels,
            feature_count,
        )

    elif features.ndim == 4:

        trials = features.shape[0]

        windows = features.shape[1]

        channels = features.shape[2]

        feature_count = features.shape[3]

        features_4d = features

    else:

        st.error(
            "Unsupported DEAP feature shape."
        )

        st.write(
            f"Received: {features.shape}"
        )

        st.stop()


    # ========================================================
    # DEAP CHANNEL NAMES
    # ========================================================

    channel_names = [
        "FP1",
        "AF3",
        "F3",
        "F7",
        "FC5",
        "FC1",
        "C3",
        "T7",
        "CP5",
        "CP1",
        "P3",
        "P7",
        "PO3",
        "O1",
        "OZ",
        "O2",
        "PO4",
        "P8",
        "P4",
        "CP6",
        "CP2",
        "C4",
        "T8",
        "FC6",
        "FC2",
        "F4",
        "F8",
        "AF4",
        "FP2",
        "FZ",
        "CZ",
        "PZ",
    ]


    # ========================================================
    # DEAP FEATURE NAMES
    # ========================================================

    feature_names = [
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


    # ========================================================
    # STATUS
    # ========================================================

    st.html(
        """
        <div class="success-card">

            <div class="success-title">
                DEAP dataset is active
            </div>

            <div class="success-text">
                The visualizations below are generated
                from the processed DEAP dataset uploaded
                through the Upload Dataset page.
            </div>

        </div>
        """
    )


    # ========================================================
    # CURRENT DATASET
    # ========================================================

    st.markdown(
        '<div class="section-title">Current Dataset</div>',
        unsafe_allow_html=True,
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Dataset",
            "DEAP",
        )


    with col2:

        st.metric(
            "Trials",
            trials,
        )


    with col3:

        st.metric(
            "Windows / Trial",
            windows,
        )


    with col4:

        st.metric(
            "EEG Channels",
            channels,
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    st.markdown(
        '<div class="section-title">Visualization Controls</div>',
        unsafe_allow_html=True,
    )


    control1, control2 = st.columns(2)


    with control1:

        selected_trial = st.selectbox(
            "Select Trial",
            options=list(
                range(
                    1,
                    trials + 1,
                )
            ),
            index=0,
            key="deap_visualization_trial",
        )


    with control2:

        selected_window = st.selectbox(
            "Select EEG Window",
            options=list(
                range(
                    1,
                    windows + 1,
                )
            ),
            index=0,
            key="deap_visualization_window",
        )


    trial_index = (
        selected_trial - 1
    )

    window_index = (
        selected_window - 1
    )


    selected_sample = features_4d[
        trial_index,
        window_index,
    ]


    # ========================================================
    # EMOTION LABELS
    # ========================================================

    st.markdown(
        '<div class="section-title">Emotion Labels</div>',
        unsafe_allow_html=True,
    )


    valence_score = None

    arousal_score = None

    valence_label = "Unknown"

    arousal_label = "Unknown"


    if (
        labels is not None
        and labels.ndim >= 2
        and trial_index < len(labels)
    ):

        trial_label = labels[
            trial_index
        ]


        if len(trial_label) >= 2:

            valence_score = float(
                trial_label[0]
            )

            arousal_score = float(
                trial_label[1]
            )


            # Project threshold
            # 5.0

            if valence_score >= 5.0:

                valence_label = "High"

            else:

                valence_label = "Low"


            if arousal_score >= 5.0:

                arousal_label = "High"

            else:

                arousal_label = "Low"


    label_col1, label_col2 = st.columns(2)


    with label_col1:

        score_text = ""

        if valence_score is not None:

            score_text = (
                f"Original DEAP score: "
                f"{valence_score:.2f}"
            )


        st.html(
            f"""
            <div class="label-card">

                <div class="label-name">
                    Valence
                </div>

                <div class="label-value">
                    {valence_label}
                </div>

                <div class="label-score">
                    {score_text}
                </div>

            </div>
            """
        )


    with label_col2:

        score_text = ""

        if arousal_score is not None:

            score_text = (
                f"Original DEAP score: "
                f"{arousal_score:.2f}"
            )


        st.html(
            f"""
            <div class="label-card">

                <div class="label-name">
                    Arousal
                </div>

                <div class="label-value">
                    {arousal_label}
                </div>

                <div class="label-score">
                    {score_text}
                </div>

            </div>
            """
        )


    # ========================================================
    # SELECTED WINDOW
    # ========================================================

    st.markdown(
        '<div class="section-title">Selected EEG Window</div>',
        unsafe_allow_html=True,
    )


    st.write(
        f"**Trial {selected_trial}  "
        f"Window {selected_window}**"
    )


    st.caption(
        "This window contains "
        "32 EEG channels × 10 features."
    )


    # ========================================================
    # DEAP HEATMAP
    # ========================================================

    st.markdown(
        '<div class="section-title">Channel × Feature Heatmap</div>',
        unsafe_allow_html=True,
    )


    heatmap_df = pd.DataFrame(
        selected_sample.T,
        index=feature_names,
        columns=channel_names,
    )


    heatmap_fig = px.imshow(
        heatmap_df,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "EEG Channel",
            "y": "Feature",
            "color": "Value",
        },
    )


    heatmap_fig.update_layout(
        title=(
            f"DEAP Trial {selected_trial} "
            f" - Window {selected_window}"
        ),
        height=520,
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50,
        ),
    )


    st.plotly_chart(
        heatmap_fig,
        use_container_width=True,
    )


    # ========================================================
    # DEAP FEATURE TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">Feature Values</div>',
        unsafe_allow_html=True,
    )


    feature_table = pd.DataFrame(
        selected_sample,
        columns=feature_names,
        index=channel_names,
    )


    feature_table.index.name = (
        "EEG Channel"
    )


    st.dataframe(
        feature_table.round(4),
        use_container_width=True,
        height=500,
    )


    # ========================================================
    # INDIVIDUAL FEATURE
    # ========================================================

    st.markdown(
        '<div class="section-title">Individual Feature Analysis</div>',
        unsafe_allow_html=True,
    )


    selected_feature = st.selectbox(
        "Select Feature",
        options=feature_names,
        key="deap_feature_selection",
    )


    feature_index = feature_names.index(
        selected_feature
    )


    feature_values = selected_sample[
        :,
        feature_index,
    ]


    feature_df = pd.DataFrame(
        {
            "Channel": channel_names,
            "Value": feature_values,
        }
    )


    feature_fig = px.bar(
        feature_df,
        x="Channel",
        y="Value",
        title=(
            f"{selected_feature} "
            "Across EEG Channels"
        ),
    )


    feature_fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=80,
        ),
    )


    st.plotly_chart(
        feature_fig,
        use_container_width=True,
    )


    # ========================================================
    # DEAP WINDOW-LEVEL ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">Window-Level Analysis</div>',
        unsafe_allow_html=True,
    )


    st.write(
        """
        This graph shows how the average feature
        magnitude changes across the windows of
        the selected DEAP trial.
        """
    )


    window_means = np.mean(
        features_4d[
            trial_index
        ],
        axis=(1, 2),
    )


    window_df = pd.DataFrame(
        {
            "Window": np.arange(
                1,
                windows + 1,
            ),
            "Mean Feature Value": window_means,
        }
    )


    window_fig = px.line(
        window_df,
        x="Window",
        y="Mean Feature Value",
        markers=True,
        title=(
            f"Trial {selected_trial} "
            "Feature Magnitude Across Windows"
        ),
    )


    window_fig.update_layout(
        height=430,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=50,
        ),
    )


    st.plotly_chart(
        window_fig,
        use_container_width=True,
    )


    # ========================================================
    # DEAP EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">What This Visualization Represents</div>',
        unsafe_allow_html=True,
    )


    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                DEAP EEG Feature Representation
            </div>

            <div class="info-text">

                Each DEAP EEG window is represented
                using 32 EEG electrodes and 10
                extracted features.

                <br><br>

                The 10 features consist of:

                <br>

                <b>Differential Entropy (DE):</b>
                Delta, Theta, Alpha, Beta and Gamma

                <br>

                <b>Power Spectral Density (PSD):</b>
                Delta, Theta, Alpha, Beta and Gamma

                <br><br>

                Therefore, each EEG window contains:

                <br><br>

                <b>
                32 channels × 10 features = 320 values
                </b>

            </div>

        </div>
        """
    )


# ============================================================
# ============================================================
# SEED VISUALIZATION
# ============================================================
# ============================================================

elif dataset_name == "SEED":

    # ========================================================
    # VALIDATE SEED
    # ========================================================

    if features.ndim != 3:

        st.error(
            "Unexpected SEED feature representation."
        )

        st.write(
            f"Received shape: {features.shape}"
        )

        st.write(
            "Expected: (samples, 5, 62)"
        )

        st.stop()


    samples = features.shape[0]

    bands = features.shape[1]

    channels = features.shape[2]


    if bands != 5:

        st.error(
            "SEED data must contain "
            "5 frequency bands."
        )

        st.write(
            f"Received: {bands} bands"
        )

        st.stop()


    if channels != 62:

        st.error(
            "SEED data must contain "
            "62 EEG channels."
        )

        st.write(
            f"Received: {channels} channels"
        )

        st.stop()


    # ========================================================
    # SEED BAND NAMES
    # ========================================================

    band_names = [
        "Delta",
        "Theta",
        "Alpha",
        "Beta",
        "Gamma",
    ]


    # ========================================================
    # SEED CHANNEL NAMES
    # ========================================================

    seed_channels = [
        "FP1",
        "FPZ",
        "FP2",
        "AF3",
        "AF4",
        "F7",
        "F5",
        "F3",
        "F1",
        "FZ",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCZ",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "CZ",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPZ",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "P7",
        "P5",
        "P3",
        "P1",
        "PZ",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO5",
        "PO3",
        "POZ",
        "PO4",
        "PO6",
        "PO8",
        "O1",
        "OZ",
        "O2",
        "CB1",
        "CB2",
    ]


    # ========================================================
    # STATUS
    # ========================================================

    st.html(
        """
        <div class="success-card">

            <div class="success-title">
                SEED dataset is active
            </div>

            <div class="success-text">
                The visualizations below are generated
                from the processed SEED dataset uploaded
                through the Upload Dataset page.
            </div>

        </div>
        """
    )


    # ========================================================
    # DATASET METRICS
    # ========================================================

    st.markdown(
        '<div class="section-title">Current Dataset</div>',
        unsafe_allow_html=True,
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Dataset",
            "SEED",
        )


    with col2:

        st.metric(
            "Samples",
            f"{samples:,}",
        )


    with col3:

        st.metric(
            "Frequency Bands",
            bands,
        )


    with col4:

        st.metric(
            "EEG Channels",
            channels,
        )


    # ========================================================
    # SUBJECT INFORMATION
    # ========================================================

    if subject_ids is not None:

        unique_subjects = np.unique(
            subject_ids
        )

        st.caption(
            f"{len(unique_subjects)} subjects "
            "available in the uploaded dataset."
        )


    # ========================================================
    # SAMPLE CONTROL
    # ========================================================

    st.markdown(
        '<div class="section-title">Visualization Controls</div>',
        unsafe_allow_html=True,
    )


    selected_sample_number = st.number_input(
        "Select EEG Sample",
        min_value=1,
        max_value=samples,
        value=1,
        step=1,
        key="seed_sample_number",
    )


    sample_index = (
        int(
            selected_sample_number
        )
        - 1
    )


    selected_sample = features[
        sample_index
    ]


    # ========================================================
    # SUBJECT FOR SELECTED SAMPLE
    # ========================================================

    selected_subject = None


    if (
        subject_ids is not None
        and sample_index < len(
            subject_ids
        )
    ):

        selected_subject = int(
            subject_ids[
                sample_index
            ]
        )


    if selected_subject is not None:

        st.caption(
            f"Selected sample belongs to "
            f"Subject {selected_subject}"
        )


    # ========================================================
    # EMOTION LABEL
    # ========================================================

    st.markdown(
        '<div class="section-title">Emotion Label</div>',
        unsafe_allow_html=True,
    )


    seed_label_names = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }


    current_label = None


    if (
        labels is not None
        and sample_index < len(
            labels
        )
    ):

        current_label = int(
            labels[
                sample_index
            ]
        )


    emotion_name = seed_label_names.get(
        current_label,
        "Unknown",
    )


    label_score = (
        f"Class ID: {current_label}"
        if current_label is not None
        else "Class ID unavailable"
    )


    st.html(
        f"""
        <div class="label-card">

            <div class="label-name">
                SEED Emotion
            </div>

            <div class="label-value">
                {emotion_name}
            </div>

            <div class="label-score">
                {label_score}
            </div>

        </div>
        """
    )


    # ========================================================
    # SEED HEATMAP
    # ========================================================

    st.markdown(
        '<div class="section-title">Channel × Frequency Band Heatmap</div>',
        unsafe_allow_html=True,
    )


    seed_heatmap = pd.DataFrame(
        selected_sample,
        index=band_names,
        columns=seed_channels,
    )


    seed_heatmap_fig = px.imshow(
        seed_heatmap,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={
            "x": "EEG Channel",
            "y": "Frequency Band",
            "color": "Feature Value",
        },
    )


    seed_heatmap_fig.update_layout(
        title=(
            f"SEED Sample "
            f"{selected_sample_number}"
        ),
        height=500,
        margin=dict(
            l=40,
            r=40,
            t=70,
            b=50,
        ),
    )


    st.plotly_chart(
        seed_heatmap_fig,
        use_container_width=True,
    )


    # ========================================================
    # SEED FEATURE TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">Frequency-Band Feature Values</div>',
        unsafe_allow_html=True,
    )


    seed_table = pd.DataFrame(
        selected_sample.T,
        columns=band_names,
        index=seed_channels,
    )


    seed_table.index.name = (
        "EEG Channel"
    )


    st.dataframe(
        seed_table.round(4),
        use_container_width=True,
        height=500,
    )


    # ========================================================
    # BAND ANALYSIS
    # ========================================================

    st.markdown(
        '<div class="section-title">Individual Frequency-Band Analysis</div>',
        unsafe_allow_html=True,
    )


    selected_band = st.selectbox(
        "Select Frequency Band",
        band_names,
        key="seed_band_selection",
    )


    band_index = band_names.index(
        selected_band
    )


    band_values = selected_sample[
        band_index
    ]


    band_df = pd.DataFrame(
        {
            "Channel": seed_channels,
            "Feature Value": band_values,
        }
    )


    band_fig = px.bar(
        band_df,
        x="Channel",
        y="Feature Value",
        title=(
            f"{selected_band} Band "
            "Across 62 EEG Channels"
        ),
    )


    band_fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        margin=dict(
            l=30,
            r=30,
            t=70,
            b=80,
        ),
    )


    st.plotly_chart(
        band_fig,
        use_container_width=True,
    )


    # ========================================================
    # BAND AVERAGES
    # ========================================================

    st.markdown(
        '<div class="section-title">Average Activity by Frequency Band</div>',
        unsafe_allow_html=True,
    )


    band_means = np.mean(
        selected_sample,
        axis=1,
    )


    band_mean_df = pd.DataFrame(
        {
            "Frequency Band": band_names,
            "Average Feature Value": band_means,
        }
    )


    band_mean_fig = px.bar(
        band_mean_df,
        x="Frequency Band",
        y="Average Feature Value",
        title=(
            f"SEED Sample "
            f"{selected_sample_number}"
        ),
    )


    band_mean_fig.update_layout(
        height=400,
    )


    st.plotly_chart(
        band_mean_fig,
        use_container_width=True,
    )


    # ========================================================
    # SEED EXPLANATION
    # ========================================================

    st.markdown(
        '<div class="section-title">What This Visualization Represents</div>',
        unsafe_allow_html=True,
    )


    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                SEED EEG Feature Representation
            </div>

            <div class="info-text">

                Each SEED sample in this project is
                represented using five frequency-band
                features across 62 EEG electrodes.

                <br><br>

                The five frequency bands are:

                <br>

                <b>
                Delta · Theta · Alpha · Beta · Gamma
                </b>

                <br><br>

                The heatmap therefore contains:

                <br><br>

                <b>
                5 frequency bands × 62 EEG channels
                </b>

                <br><br>

                Each cell represents the feature value
                for one frequency band at one EEG
                electrode.

            </div>

        </div>
        """
    )


    # ========================================================
    # SEED SPATIAL INFORMATION
    # ========================================================

    col1, col2 = st.columns(2)


    with col1:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    Spatial Representation
                </div>

                <div class="info-text">

                    The 62 EEG channels represent
                    different electrode locations
                    across the scalp.

                    <br><br>

                    These electrode relationships
                    are later represented as a graph
                    for the GCN model.

                </div>

            </div>
            """
        )


    with col2:

        st.html(
            """
            <div class="info-card">

                <div class="info-title">
                    Frequency Representation
                </div>

                <div class="info-text">

                    The five frequency bands provide
                    the feature representation used
                    by the SEED spatial learning
                    pipeline.

                    <br><br>

                    The visualization allows you to
                    inspect how feature values vary
                    across the EEG electrodes.

                </div>

            </div>
            """
        )


# ============================================================
# UNKNOWN DATASET
# ============================================================

else:

    st.error(
        f"Unsupported dataset: {dataset_type}"
    )

    st.info(
        "The dashboard currently supports "
        "DEAP and SEED."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        EEG based Emotion Recognition using
        Spatial-Temporal Representation Learning
        · DEAP + SEED
    </div>
    """,
    unsafe_allow_html=True,
)