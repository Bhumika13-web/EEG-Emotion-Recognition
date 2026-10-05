import streamlit as st
import numpy as np
from pathlib import Path
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Emotion Prediction",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

SEED_EMOTION_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

DEAP_VALENCE_THRESHOLD = 5.0
DEAP_AROUSAL_THRESHOLD = 5.0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

DEAP_VALENCE_MODEL = (
    MODEL_DIR / "svm_valence_best.joblib"
)

DEAP_AROUSAL_MODEL = (
    MODEL_DIR / "svm_arousal_best.joblib"
)

SEED_MODEL = (
    MODEL_DIR / "svm_seed.joblib"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

.page-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 6px;
}

.page-subtitle {
    font-size: 16px;
    opacity: 0.70;
    margin-bottom: 28px;
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    margin-top: 28px;
    margin-bottom: 14px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid var(--st-border-color);
    background: var(--st-secondary-background-color);
    margin-top: 10px;
    margin-bottom: 10px;
}

.info-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 8px;
}

.small-text {
    font-size: 14px;
    opacity: 0.75;
    line-height: 1.7;
}

.result-box {
    padding: 22px;
    border-radius: 14px;
    border: 1px solid var(--st-border-color);
    background: var(--st-secondary-background-color);
    text-align: center;
    min-height: 145px;
}

.result-label {
    font-size: 14px;
    opacity: 0.65;
    margin-bottom: 8px;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
}

.result-description {
    font-size: 13px;
    opacity: 0.65;
    margin-top: 8px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

features = st.session_state.get(
    "processed_features"
)

labels = st.session_state.get(
    "processed_labels"
)

dataset_type = st.session_state.get(
    "dataset_type"
)

dataset_processed = st.session_state.get(
    "dataset_processed",
    False
)


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):

    path = Path(model_path)

    if not path.exists():
        return None

    return joblib.load(path)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_deap_features(data):
    """
    Expected final representation:

    trials × windows × channels × features

    Example:

    (40, 30, 32, 10)

    or

    (40, 30, 320)
    """

    data = np.asarray(data)

    if data.ndim == 4:

        if (
            data.shape[2] == 32
            and data.shape[3] == 10
        ):
            return data

    if data.ndim == 3:

        if data.shape[-1] == 320:

            return data.reshape(
                data.shape[0],
                data.shape[1],
                32,
                10
            )

    raise ValueError(
        f"Unsupported DEAP feature shape: {data.shape}"
    )


def get_deap_label(
    label_array,
    trial_index
):

    if label_array is None:
        return None

    labels_np = np.asarray(
        label_array
    )

    if labels_np.ndim != 2:
        return None

    if trial_index >= labels_np.shape[0]:
        return None

    if labels_np.shape[1] < 2:
        return None

    valence = float(
        labels_np[
            trial_index,
            0
        ]
    )

    arousal = float(
        labels_np[
            trial_index,
            1
        ]
    )

    return {
        "valence_score": valence,
        "arousal_score": arousal,

        "valence_class": (
            "High"
            if valence >= DEAP_VALENCE_THRESHOLD
            else "Low"
        ),

        "arousal_class": (
            "High"
            if arousal >= DEAP_AROUSAL_THRESHOLD
            else "Low"
        ),
    }


def get_seed_label(
    label_array,
    sample_index
):

    if label_array is None:
        return None

    labels_np = np.asarray(
        label_array
    ).reshape(-1)

    if sample_index >= len(labels_np):
        return None

    class_id = int(
        labels_np[sample_index]
    )

    emotion = SEED_EMOTION_MAP.get(
        class_id
    )

    if emotion is None:
        return None

    return {
        "class_id": class_id,
        "emotion": emotion
    }


def prepare_deap_svm_input(
    trial_features
):

    data = np.asarray(
        trial_features
    )

    if data.ndim == 3:

        data = data.reshape(
            data.shape[0],
            -1
        )

    if data.ndim != 2:

        raise ValueError(
            f"Unexpected DEAP shape: {data.shape}"
        )

    temporal_mean = np.mean(
        data,
        axis=0
    )

    return temporal_mean.reshape(
        1,
        -1
    )


def prepare_seed_svm_input(
    sample
):

    data = np.asarray(
        sample
    )

    if data.shape != (5, 62):

        raise ValueError(
            f"Unexpected SEED sample shape: {data.shape}"
        )

    return data.reshape(
        1,
        -1
    )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">Emotion Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="page-subtitle">
        Predict emotional state from the processed EEG representation
        and compare the prediction with the reference dataset label.
    </div>
    """,
    unsafe_allow_html=True
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
        "No processed dataset is available. "
        "Please open Upload Dataset and process "
        "a DEAP or SEED dataset."
    )

    st.stop()


# ============================================================
# CURRENT DATASET
# ============================================================

st.markdown(
    '<div class="section-title">Current Dataset</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset",
        dataset_type
    )

with col2:

    st.metric(
        "Samples / Trials",
        f"{len(features):,}"
    )

with col3:

    if dataset_type == "DEAP":

        representation = "30 × 32 × 10"

    else:

        representation = "5 × 62"

    st.metric(
        "Representation",
        representation
    )


# ============================================================
# DEAP
# ============================================================

if dataset_type == "DEAP":

    # --------------------------------------------------------
    # PREPARE DEAP
    # --------------------------------------------------------

    try:

        deap_features = normalize_deap_features(
            features
        )

    except Exception as error:

        st.error(
            f"Unable to read DEAP representation: {error}"
        )

        st.stop()


    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction Controls</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        trial_number = st.selectbox(
            "Select Trial",
            range(
                1,
                deap_features.shape[0] + 1
            ),
            key="deap_prediction_trial"
        )

    with col2:

        window_number = st.selectbox(
            "Select EEG Window",
            range(
                1,
                deap_features.shape[1] + 1
            ),
            key="deap_prediction_window"
        )


    trial_index = trial_number - 1

    window_index = window_number - 1

    selected_trial = deap_features[
        trial_index
    ]

    selected_window = selected_trial[
        window_index
    ]


    # --------------------------------------------------------
    # REFERENCE LABEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Reference Emotion Label</div>',
        unsafe_allow_html=True
    )

    deap_label = get_deap_label(
        labels,
        trial_index
    )

    if deap_label is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Valence",
                deap_label["valence_class"],
                f"Score: {deap_label['valence_score']:.2f}"
            )

        with col2:

            st.metric(
                "Arousal",
                deap_label["arousal_class"],
                f"Score: {deap_label['arousal_score']:.2f}"
            )

    else:

        st.warning(
            "DEAP reference labels are not available."
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Prediction</div>',
        unsafe_allow_html=True
    )

    valence_model = load_model(
        str(DEAP_VALENCE_MODEL)
    )

    arousal_model = load_model(
        str(DEAP_AROUSAL_MODEL)
    )


    predicted_valence = None
    predicted_arousal = None


    try:

        model_input = prepare_deap_svm_input(
            selected_trial
        )


        if valence_model is not None:

            predicted_valence = int(
                valence_model.predict(
                    model_input
                )[0]
            )


        if arousal_model is not None:

            predicted_arousal = int(
                arousal_model.predict(
                    model_input
                )[0]
            )

    except Exception as error:

        st.error(
            f"Prediction error: {error}"
        )


    col1, col2 = st.columns(2)

    with col1:

        if predicted_valence is not None:

            value = (
                "High"
                if predicted_valence == 1
                else "Low"
            )

            st.metric(
                "Predicted Valence",
                value,
                "SVM baseline"
            )

        else:

            st.info(
                "Valence model is unavailable."
            )


    with col2:

        if predicted_arousal is not None:

            value = (
                "High"
                if predicted_arousal == 1
                else "Low"
            )

            st.metric(
                "Predicted Arousal",
                value,
                "SVM baseline"
            )

        else:

            st.info(
                "Arousal model is unavailable."
            )


    # --------------------------------------------------------
    # INPUT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Selected EEG Input</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Trial",
            trial_number
        )

    with col2:

        st.metric(
            "EEG Channels",
            32
        )

    with col3:

        st.metric(
            "Features / Channel",
            10
        )


    st.markdown(
        f"""
        <div class="info-box">

        <div class="info-title">
        Selected EEG Window
        </div>

        <div class="small-text">

        Trial: <b>{trial_number}</b><br>
        Window: <b>{window_number}</b><br>
        Input shape: <b>{selected_window.shape}</b><br>
        Representation:
        <b>Differential Entropy + PSD</b>

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )

    st.info(
        """
        DEAP prediction uses the trained SVM baseline models.

        Input:
        30 temporal windows × 32 EEG channels × 10 features.

        Features:
        Differential Entropy + PSD.

        Classification:
        Binary Valence and Arousal.

        Threshold:
        5.0.
        """
    )


# ============================================================
# SEED
# ============================================================

elif dataset_type == "SEED":

    seed_features = np.asarray(
        features
    )


    # --------------------------------------------------------
    # VALIDATE SEED
    # --------------------------------------------------------

    if seed_features.ndim != 3:

        st.error(
            "Expected SEED representation "
            "(samples, 5, 62). "
            f"Received {seed_features.shape}."
        )

        st.stop()


    if seed_features.shape[1:] != (5, 62):

        st.error(
            "Expected SEED representation "
            "(samples, 5, 62). "
            f"Received {seed_features.shape}."
        )

        st.stop()


    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Prediction Controls</div>',
        unsafe_allow_html=True
    )

    sample_number = st.selectbox(
        "Select SEED Sample",
        range(
            1,
            len(seed_features) + 1
        ),
        key="seed_prediction_sample"
    )


    sample_index = sample_number - 1

    selected_sample = seed_features[
        sample_index
    ]


    # --------------------------------------------------------
    # REFERENCE LABEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Reference Emotion Label</div>',
        unsafe_allow_html=True
    )

    seed_label = get_seed_label(
        labels,
        sample_index
    )


    if seed_label is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "SEED Emotion",
                seed_label["emotion"]
            )

        with col2:

            st.metric(
                "Class ID",
                seed_label["class_id"]
            )

    else:

        st.warning(
            """
            SEED reference label is unavailable.

            The feature data is loaded, but the matching
            LabelsNoImage.npz data is not available in
            the current session.

            Please return to Upload Dataset and process:

            DatasetCaricatoNoImage.npz
            LabelsNoImage.npz
            SubjectsNoImage.npz
            """
        )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Prediction</div>',
        unsafe_allow_html=True
    )


    seed_model = load_model(
        str(SEED_MODEL)
    )


    predicted_class = None
    predicted_emotion = None


    if seed_model is not None:

        try:

            model_input = prepare_seed_svm_input(
                selected_sample
            )

            predicted_class = int(
                seed_model.predict(
                    model_input
                )[0]
            )

            predicted_emotion = (
                SEED_EMOTION_MAP.get(
                    predicted_class,
                    "Unknown"
                )
            )

        except Exception as error:

            st.error(
                f"SEED prediction failed: {error}"
            )

    else:

        st.warning(
            "SEED SVM model was not found."
        )


    if predicted_emotion is not None:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Emotion",
                predicted_emotion
            )

        with col2:

            st.metric(
                "Predicted Class ID",
                predicted_class
            )


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if (
        seed_label is not None
        and predicted_class is not None
    ):

        st.markdown(
            '<div class="section-title">Prediction Comparison</div>',
            unsafe_allow_html=True
        )


        if predicted_class == seed_label["class_id"]:

            st.success(
                "Prediction matches the reference emotion."
            )

        else:

            st.warning(
                "Prediction does not match the reference emotion."
            )


    # --------------------------------------------------------
    # SELECTED INPUT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Selected EEG Input</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Sample",
            sample_number
        )

    with col2:

        st.metric(
            "Frequency Bands",
            5
        )

    with col3:

        st.metric(
            "EEG Channels",
            62
        )


    st.markdown(
        """
        <div class="info-box">

        <div class="info-title">
        SEED Representation
        </div>

        <div class="small-text">

        The selected SEED sample contains
        <b>5 frequency-band features</b>
        across
        <b>62 EEG channels</b>.

        <br><br>

        Frequency bands:
        <b>Delta, Theta, Alpha, Beta, Gamma</b>

        <br><br>

        Input shape:
        <b>(5, 62)</b>

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # EMOTION CLASSES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">SEED Emotion Classes</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Class 0",
            "Negative"
        )

    with col2:

        st.metric(
            "Class 1",
            "Neutral"
        )

    with col3:

        st.metric(
            "Class 2",
            "Positive"
        )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        SEED prediction uses the trained SVM baseline model.

        Input:
        5 frequency bands × 62 EEG channels.

        Classes:
        Negative / Neutral / Positive.

        Class mapping:
        0 = Negative,
        1 = Neutral,
        2 = Positive.

        Model:
        RBF SVM.
        """
    )


# ============================================================
# INVALID DATASET
# ============================================================

else:

    st.error(
        f"Unsupported dataset type: {dataset_type}"
    )
# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

import pandas as pd

if dataset_type == "DEAP" and 'predicted_valence' in locals() and predicted_valence is not None:
    results_df = pd.DataFrame([{
        "Trial": trial_number,
        "Window": window_number,
        "Predicted_Valence": "High" if predicted_valence == 1 else "Low",
        "Predicted_Arousal": "High" if predicted_arousal == 1 else "Low",
        "Actual_Valence": "High" if deap_label["valence"] > 5 else "Low",
        "Actual_Arousal": "High" if deap_label["arousal"] > 5 else "Low"
    }])
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download DEAP Prediction (CSV)", csv, "deap_prediction.csv", "text/csv")
    
elif dataset_type == "SEED" and 'predicted_class' in locals() and predicted_class is not None:
    results_df = pd.DataFrame([{
        "Sample": sample_number,
        "Predicted_Class": predicted_class,
        "Predicted_Emotion": predicted_emotion,
        "Actual_Class": seed_label["class_id"],
        "Actual_Emotion": seed_label["emotion"]
    }])
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download SEED Prediction (CSV)", csv, "seed_prediction.csv", "text/csv")

