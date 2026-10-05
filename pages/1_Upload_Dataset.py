import streamlit as st
import numpy as np
import pickle
import tempfile
import zipfile
from pathlib import Path

from scipy.signal import butter, sosfiltfilt, welch


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

DEAP_FS = 128
DEAP_EEG_CHANNELS = 32

DEAP_WINDOW_SECONDS = 4
DEAP_OVERLAP_SECONDS = 2

DEAP_WINDOW_SAMPLES = (
    DEAP_FS * DEAP_WINDOW_SECONDS
)

DEAP_STEP_SAMPLES = (
    DEAP_FS
    * (
        DEAP_WINDOW_SECONDS
        - DEAP_OVERLAP_SECONDS
    )
)

DEAP_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 45),
}

SEED_BANDS = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma",
]


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

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

.section-title {
    font-size: 24px;
    font-weight: 750;
    color: var(--st-heading-color) !important;
    margin-top: 30px;
    margin-bottom: 15px;
}

.info-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.info-title {
    color: var(--st-heading-color) !important;
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 10px;
}

.info-text {
    color: var(--st-text-color) !important;
    opacity: 0.82;
    line-height: 1.8;
}

.dataset-card {
    background: var(--st-secondary-background-color);
    border: 1px solid var(--st-border-color);
    border-radius: 16px;
    padding: 22px;
    min-height: 225px;
}

.dataset-title {
    color: var(--st-heading-color) !important;
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 12px;
}

.dataset-text {
    color: var(--st-text-color) !important;
    opacity: 0.82;
    line-height: 1.8;
}

.success-card {
    background: rgba(34, 197, 94, 0.10);
    border: 1px solid rgba(34, 197, 94, 0.30);
    border-radius: 15px;
    padding: 18px 22px;
    margin: 18px 0;
}

.success-title {
    color: #16a34a !important;
    font-size: 17px;
    font-weight: 750;
}

.success-text {
    color: var(--st-text-color) !important;
    opacity: 0.82;
}

.warning-card {
    background: rgba(245, 158, 11, 0.10);
    border: 1px solid rgba(245, 158, 11, 0.30);
    border-radius: 15px;
    padding: 18px 22px;
    margin: 18px 0;
}

[data-testid="stMetricLabel"] {
    color: var(--st-text-color) !important;
}

[data-testid="stMetricValue"] {
    color: var(--st-heading-color) !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "processed_features": None,
    "processed_labels": None,
    "processed_subject_ids": None,
    "dataset_type": None,
    "dataset_processed": False,
    "processed_source": None,
    "processed_filename": None,
    "upload_dataset_choice": "DEAP",
}


for key, default_value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">Upload Dataset</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-subtitle">
        Upload a DEAP or SEED dataset and prepare it for
        EEG visualization, feature analysis, and emotion recognition.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SUPPORTED DATASETS
# ============================================================

st.markdown(
    '<div class="section-title">Supported Datasets</div>',
    unsafe_allow_html=True,
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

                <b>Input:</b> .dat subject files<br>

                <b>EEG Channels:</b> 32<br>

                <b>Sampling Rate:</b> 128 Hz<br>

                <b>Trials:</b> 40 per subject<br>

                <b>Processing:</b>
                Filtering + windowing + DE + PSD<br>

                <b>Representation:</b>
                30 windows  32 channels  10 features

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

                <b>Input:</b> .npz files<br>

                <b>EEG Channels:</b> 62<br>

                <b>Frequency Bands:</b> 5<br>

                <b>Classes:</b>
                Negative / Neutral / Positive<br>

                <b>Processing:</b>
                Validation + normalization<br>

                <b>Representation:</b>
                5 frequency bands  62 channels

            </div>

        </div>
        """
    )


# ============================================================
# DATASET SELECTION
# ============================================================

st.markdown(
    '<div class="section-title">Step 1  Select Dataset</div>',
    unsafe_allow_html=True,
)

dataset_choice = st.radio(
    "Choose the dataset you want to process",
    ["DEAP", "SEED"],
    horizontal=True,
    key="dataset_selector",
)


# ============================================================
# CLEAR OLD DATA WHEN DATASET CHANGES
# ============================================================

previous_choice = st.session_state.get(
    "upload_dataset_choice"
)

if (
    previous_choice is not None
    and previous_choice != dataset_choice
):

    st.session_state.processed_features = None
    st.session_state.processed_labels = None
    st.session_state.processed_subject_ids = None
    st.session_state.dataset_type = None
    st.session_state.dataset_processed = False
    st.session_state.processed_source = None
    st.session_state.processed_filename = None


st.session_state.upload_dataset_choice = dataset_choice


# ============================================================
# PRECOMPUTE DEAP FILTERS
# ============================================================

@st.cache_resource
def get_deap_band_filters():
    """
    Create the five DEAP Butterworth filters once.

    This prevents recreating filter coefficients for
    every EEG window.
    """

    filters = {}

    nyquist = DEAP_FS / 2

    for band_name, (
        low_frequency,
        high_frequency,
    ) in DEAP_BANDS.items():

        high_frequency = min(
            high_frequency,
            nyquist - 1,
        )

        sos = butter(
            4,
            [
                low_frequency / nyquist,
                high_frequency / nyquist,
            ],
            btype="bandpass",
            output="sos",
        )

        filters[band_name] = sos

    return filters


# ============================================================
# DEAP FEATURE EXTRACTION
# ============================================================

def calculate_deap_window_features(
    window,
):
    """
    Calculate DE + PSD for ONE DEAP EEG window.

    Input:
        channels  samples

    Output:
        channels  10

    5 DE + 5 PSD
    """

    band_filters = get_deap_band_filters()

    all_de = []
    all_psd = []

    # --------------------------------------------------------
    # PSD calculated once for the window
    # --------------------------------------------------------

    frequencies, psd = welch(
        window,
        fs=DEAP_FS,
        nperseg=min(
            256,
            window.shape[-1],
        ),
        axis=-1,
    )

    # --------------------------------------------------------
    # Process each frequency band
    # --------------------------------------------------------

    for band_name, (
        low_frequency,
        high_frequency,
    ) in DEAP_BANDS.items():

        sos = band_filters[
            band_name
        ]

        band_signal = sosfiltfilt(
            sos,
            window,
            axis=-1,
        )

        # ====================================================
        # Differential Entropy
        # ====================================================

        variance = np.var(
            band_signal,
            axis=-1,
        )

        variance = np.maximum(
            variance,
            1e-12,
        )

        de_values = (
            0.5
            * np.log(
                2
                * np.pi
                * np.e
                * variance
            )
        )

        all_de.append(
            de_values
        )

        # ====================================================
        # PSD
        # ====================================================

        mask = (
            (frequencies >= low_frequency)
            & (
                frequencies
                < high_frequency
            )
        )

        if np.any(mask):

            band_psd = np.mean(
                psd[
                    :,
                    mask,
                ],
                axis=-1,
            )

        else:

            band_psd = np.zeros(
                window.shape[0],
                dtype=np.float64,
            )

        band_psd = np.log1p(
            np.maximum(
                band_psd,
                0,
            )
        )

        all_psd.append(
            band_psd
        )


    # ========================================================
    # STACK FEATURES
    # ========================================================

    de_features = np.stack(
        all_de,
        axis=-1,
    )

    psd_features = np.stack(
        all_psd,
        axis=-1,
    )

    combined = np.concatenate(
        [
            de_features,
            psd_features,
        ],
        axis=-1,
    )

    return combined.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE DEAP SUBJECT
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=32,
)
def process_deap_subject(
    file_bytes,
):
    """
    Process one DEAP subject.

    IMPORTANT:
    This function is cached.

    Therefore if Streamlit reruns the page,
    the same subject file will not be processed
    again.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".dat",
        delete=False,
    ) as temp_file:

        temp_file.write(
            file_bytes
        )

        temp_path = temp_file.name


    try:

        with open(
            temp_path,
            "rb",
        ) as file:

            subject_data = pickle.load(
                file,
                encoding="latin1",
            )

    finally:

        Path(
            temp_path
        ).unlink(
            missing_ok=True
        )


    if "data" not in subject_data:

        raise ValueError(
            "DEAP file does not contain "
            "'data'."
        )

    if "labels" not in subject_data:

        raise ValueError(
            "DEAP file does not contain "
            "'labels'."
        )


    raw_data = np.asarray(
        subject_data["data"]
    )

    labels = np.asarray(
        subject_data["labels"]
    )


    if raw_data.ndim != 3:

        raise ValueError(
            "Unexpected DEAP data shape: "
            f"{raw_data.shape}"
        )


    trials = raw_data.shape[0]

    total_channels = raw_data.shape[1]

    total_samples = raw_data.shape[2]


    if total_channels < 32:

        raise ValueError(
            "DEAP dataset must contain "
            "at least 32 EEG channels."
        )


    if total_samples < DEAP_WINDOW_SAMPLES:

        raise ValueError(
            "DEAP trial is too short for "
            "a 4-second window."
        )


    # --------------------------------------------------------
    # Use first 32 EEG channels
    # --------------------------------------------------------

    eeg_data = raw_data[
        :,
        :32,
        :,
    ]


    processed_trials = []


    # --------------------------------------------------------
    # Process trials
    # --------------------------------------------------------

    for trial_index in range(
        trials
    ):

        trial = eeg_data[
            trial_index
        ]


        trial_windows = []


        start = 0


        while (
            start
            + DEAP_WINDOW_SAMPLES
            <= total_samples
        ):

            end = (
                start
                + DEAP_WINDOW_SAMPLES
            )


            window = trial[
                :,
                start:end,
            ]


            window_features = (
                calculate_deap_window_features(
                    window
                )
            )


            # 32  10  320

            flattened = (
                window_features
                .reshape(
                    32 * 10
                )
            )


            trial_windows.append(
                flattened
            )


            start += DEAP_STEP_SAMPLES


        processed_trials.append(
            np.stack(
                trial_windows,
                axis=0,
            )
        )


    features = np.stack(
        processed_trials,
        axis=0,
    )


    return (
        features.astype(
            np.float32
        ),
        labels.astype(
            np.float32
        ),
    )


# ============================================================
# FIND DEAP FILES
# ============================================================

@st.cache_data(show_spinner=False)
def extract_deap_files(
    uploaded_files,
):
    """
    Find individual .dat files.

    ZIP files are also supported if they contain
    DEAP .dat files.
    """

    result = []


    for uploaded_file in uploaded_files:

        filename = (
            uploaded_file.name
            .lower()
        )


        # ----------------------------------------------------
        # Direct .dat
        # ----------------------------------------------------

        if filename.endswith(
            ".dat"
        ):

            result.append(
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
            )


        # ----------------------------------------------------
        # ZIP
        # ----------------------------------------------------

        elif filename.endswith(
            ".zip"
        ):

            try:

                zip_bytes = (
                    uploaded_file.getvalue()
                )


                with zipfile.ZipFile(
                    tempfile.SpooledTemporaryFile(
                        max_size=50 * 1024 * 1024
                    ),
                    "r",
                ) as _:

                    pass

            except Exception:

                # Use direct BytesIO fallback

                import io

                with zipfile.ZipFile(
                    io.BytesIO(
                        zip_bytes
                    ),
                    "r",
                ) as archive:

                    for info in archive.infolist():

                        if (
                            not info.is_dir()
                            and info.filename
                            .lower()
                            .endswith(
                                ".dat"
                            )
                        ):

                            result.append(
                                (
                                    Path(
                                        info.filename
                                    ).name,
                                    archive.read(
                                        info
                                    ),
                                )
                            )

    return result


# ============================================================
# SEED LOADING
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=4,
)
def load_seed_arrays(
    data_bytes,
    labels_bytes,
    subjects_bytes,
):
    """
    Load the three SEED NPZ files.

    This function is cached, so repeated Streamlit
    reruns do not repeatedly read and validate
    the same dataset.
    """

    import io


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data_npz = np.load(
        io.BytesIO(
            data_bytes
        ),
        allow_pickle=True,
    )


    if len(
        data_npz.files
    ) == 0:

        raise ValueError(
            "Dataset NPZ contains no arrays."
        )


    data_array = np.asarray(
        data_npz[
            data_npz.files[0]
        ]
    )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels_array = None


    if labels_bytes is not None:

        labels_npz = np.load(
            io.BytesIO(
                labels_bytes
            ),
            allow_pickle=True,
        )


        if len(
            labels_npz.files
        ) > 0:

            labels_array = np.asarray(
                labels_npz[
                    labels_npz.files[0]
                ]
            )


    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    subjects_array = None


    if subjects_bytes is not None:

        subjects_npz = np.load(
            io.BytesIO(
                subjects_bytes
            ),
            allow_pickle=True,
        )


        if len(
            subjects_npz.files
        ) > 0:

            subjects_array = np.asarray(
                subjects_npz[
                    subjects_npz.files[0]
                ]
            )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if data_array.ndim != 3:

        raise ValueError(
            "SEED data must have 3 dimensions."
        )


    if data_array.shape[1] != 5:

        raise ValueError(
            "SEED data must contain "
            "5 frequency bands."
        )


    if data_array.shape[2] != 62:

        raise ValueError(
            "SEED data must contain "
            "62 EEG channels."
        )


    data_array = data_array.astype(
        np.float32
    )


    if not np.isfinite(
        data_array
    ).all():

        raise ValueError(
            "SEED data contains NaN "
            "or infinite values."
        )


    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    if labels_array is not None:

        labels_array = (
            labels_array
            .reshape(-1)
            .astype(
                np.int64
            )
        )


        if (
            len(labels_array)
            != data_array.shape[0]
        ):

            raise ValueError(
                "Number of SEED labels "
                "does not match samples."
            )


    # --------------------------------------------------------
    # SUBJECTS
    # --------------------------------------------------------

    if subjects_array is not None:

        subjects_array = (
            subjects_array
            .reshape(-1)
            .astype(
                np.int32
            )
        )


        if (
            len(subjects_array)
            != data_array.shape[0]
        ):

            raise ValueError(
                "Number of SEED subject IDs "
                "does not match samples."
            )


    return (
        data_array,
        labels_array,
        subjects_array,
    )


# ============================================================
# DEAP SECTION
# ============================================================

if dataset_choice == "DEAP":

    st.markdown(
        '<div class="section-title">Step 2  Upload DEAP Dataset</div>',
        unsafe_allow_html=True,
    )


    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                DEAP Upload
            </div>

            <div class="info-text">

                Upload one or more DEAP subject files
                such as <b>s01.dat</b>, <b>s02.dat</b>, etc.

                <br><br>

                You can also upload a ZIP containing
                individual DEAP <b>.dat</b> files.

                <br><br>

                <b>Recommended:</b>
                Upload individual subject files instead
                of the complete 2.7 GB DEAP archive.

                <br><br>

                The application processes each subject
                only when you click the
                <b>Process DEAP Dataset</b> button.

            </div>

        </div>
        """
    )


    uploaded_deap_files = st.file_uploader(
        "Upload DEAP .dat or ZIP file",
        type=[
            "dat",
            "zip",
        ],
        accept_multiple_files=True,
        key="deap_upload_files",
    )


    if uploaded_deap_files:

        st.markdown(
            '<div class="section-title">Uploaded Files</div>',
            unsafe_allow_html=True,
        )


        for file in uploaded_deap_files:

            size_mb = (
                file.size
                / (
                    1024 * 1024
                )
            )

            st.write(
                f" {file.name} "
                f" {size_mb:.1f} MB"
            )


        st.markdown(
            '<div class="section-title">Processing</div>',
            unsafe_allow_html=True,
        )


        process_deap = st.button(
            " Process DEAP Dataset",
            type="primary",
            use_container_width=True,
            key="process_deap_button",
        )


        if process_deap:

            with st.status(
                "Processing DEAP dataset...",
                expanded=True,
            ) as status:

                try:

                    dat_files = (
                        extract_deap_files(
                            uploaded_deap_files
                        )
                    )


                    if not dat_files:

                        raise ValueError(
                            "No DEAP .dat files "
                            "were found."
                        )


                    st.write(
                        f"Found {len(dat_files)} "
                        f"DEAP subject file(s)."
                    )


                    all_features = []

                    all_labels = []

                    processed_names = []


                    progress = st.progress(
                        0
                    )


                    for index, (
                        filename,
                        file_bytes,
                    ) in enumerate(
                        dat_files
                    ):

                        st.write(
                            f"Processing "
                            f"{filename}..."
                        )


                        subject_features, subject_labels = (
                            process_deap_subject(
                                file_bytes
                            )
                        )


                        all_features.append(
                            subject_features
                        )

                        all_labels.append(
                            subject_labels
                        )

                        processed_names.append(
                            filename
                        )


                        progress.progress(
                            (
                                index + 1
                            )
                            / len(
                                dat_files
                            )
                        )


                    # ----------------------------------------
                    # COMBINE
                    # ----------------------------------------

                    features = np.concatenate(
                        all_features,
                        axis=0,
                    )


                    labels = np.concatenate(
                        all_labels,
                        axis=0,
                    )


                    # ----------------------------------------
                    # SESSION STATE
                    # ----------------------------------------

                    st.session_state.processed_features = (
                        features
                    )

                    st.session_state.processed_labels = (
                        labels
                    )

                    st.session_state.processed_subject_ids = (
                        None
                    )

                    st.session_state.dataset_type = (
                        "DEAP"
                    )

                    st.session_state.dataset_processed = (
                        True
                    )

                    st.session_state.processed_source = (
                        "Uploaded DEAP dataset"
                    )

                    st.session_state.processed_filename = (
                        ", ".join(
                            processed_names
                        )
                    )


                    status.update(
                        label="DEAP processing complete!",
                        state="complete",
                        expanded=False,
                    )


                    # ----------------------------------------
                    # RESULTS
                    # ----------------------------------------

                    st.markdown(
                        '<div class="section-title">Processing Complete</div>',
                        unsafe_allow_html=True,
                    )


                    c1, c2, c3, c4 = st.columns(
                        4
                    )


                    with c1:

                        st.metric(
                            "Trials",
                            features.shape[0],
                        )


                    with c2:

                        st.metric(
                            "Windows / Trial",
                            features.shape[1],
                        )


                    with c3:

                        st.metric(
                            "EEG Channels",
                            32,
                        )


                    with c4:

                        st.metric(
                            "Features / Channel",
                            10,
                        )


                    st.success(
                        " DEAP dataset is ready."
                    )


                    st.write(
                        "Processed representation:"
                    )


                    st.code(
                        str(
                            features.shape
                        )
                    )


                except Exception as error:

                    status.update(
                        label="DEAP processing failed",
                        state="error",
                        expanded=True,
                    )

                    st.error(
                        f"DEAP processing error: "
                        f"{error}"
                    )


# ============================================================
# SEED SECTION
# ============================================================

elif dataset_choice == "SEED":

    st.markdown(
        '<div class="section-title">Step 2  Upload SEED Dataset</div>',
        unsafe_allow_html=True,
    )


    st.html(
        """
        <div class="info-card">

            <div class="info-title">
                SEED Upload
            </div>

            <div class="info-text">

                Upload the three SEED files:

                <br><br>

                <b>DatasetCaricatoNoImage.npz</b>
                 EEG feature data

                <br>

                <b>LabelsNoImage.npz</b>
                 emotion labels

                <br>

                <b>SubjectsNoImage.npz</b>
                 subject IDs

                <br><br>

                Expected representation:

                <br><br>

                <b>
                samples  5 frequency bands  62 channels
                </b>

            </div>

        </div>
        """
    )


    uploaded_seed_files = st.file_uploader(
        "Upload SEED .npz files",
        type=[
            "npz",
        ],
        accept_multiple_files=True,
        key="seed_upload_files",
    )


    if uploaded_seed_files:

        st.markdown(
            '<div class="section-title">Uploaded Files</div>',
            unsafe_allow_html=True,
        )


        for file in uploaded_seed_files:

            size_mb = (
                file.size
                / (
                    1024 * 1024
                )
            )

            st.write(
                f" {file.name} "
                f" {size_mb:.1f} MB"
            )


        st.markdown(
            '<div class="section-title">Processing</div>',
            unsafe_allow_html=True,
        )


        process_seed = st.button(
            " Process SEED Dataset",
            type="primary",
            use_container_width=True,
            key="process_seed_button",
        )


        if process_seed:

            with st.status(
                "Loading SEED dataset...",
                expanded=True,
            ) as status:

                try:

                    data_file = None
                    labels_file = None
                    subjects_file = None


                    # ----------------------------------------
                    # IDENTIFY FILES BY NAME
                    # ----------------------------------------

                    for file in uploaded_seed_files:

                        filename = (
                            file.name.lower()
                        )


                        if (
                            "dataset"
                            in filename
                        ):

                            data_file = file


                        elif (
                            "label"
                            in filename
                        ):

                            labels_file = file


                        elif (
                            "subject"
                            in filename
                        ):

                            subjects_file = file


                    # ----------------------------------------
                    # FALLBACK
                    # ----------------------------------------

                    if (
                        data_file is None
                        and len(
                            uploaded_seed_files
                        ) >= 1
                    ):

                        data_file = (
                            uploaded_seed_files[0]
                        )


                    if (
                        labels_file is None
                        and len(
                            uploaded_seed_files
                        ) >= 2
                    ):

                        labels_file = (
                            uploaded_seed_files[1]
                        )


                    if (
                        subjects_file is None
                        and len(
                            uploaded_seed_files
                        ) >= 3
                    ):

                        subjects_file = (
                            uploaded_seed_files[2]
                        )


                    if data_file is None:

                        raise ValueError(
                            "SEED dataset file "
                            "was not found."
                        )


                    st.write(
                        "Loading SEED feature data..."
                    )


                    data_bytes = (
                        data_file.getvalue()
                    )


                    labels_bytes = (
                        labels_file.getvalue()
                        if labels_file is not None
                        else None
                    )


                    subjects_bytes = (
                        subjects_file.getvalue()
                        if subjects_file is not None
                        else None
                    )


                    (
                        data_array,
                        labels_array,
                        subjects_array,
                    ) = load_seed_arrays(
                        data_bytes,
                        labels_bytes,
                        subjects_bytes,
                    )


                    # ----------------------------------------
                    # SESSION STATE
                    # ----------------------------------------

                    st.session_state.processed_features = (
                        data_array
                    )

                    st.session_state.processed_labels = (
                        labels_array
                    )

                    st.session_state.processed_subject_ids = (
                        subjects_array
                    )

                    st.session_state.dataset_type = (
                        "SEED"
                    )

                    st.session_state.dataset_processed = (
                        True
                    )

                    st.session_state.processed_source = (
                        "Uploaded SEED dataset"
                    )

                    st.session_state.processed_filename = (
                        ", ".join(
                            file.name
                            for file
                            in uploaded_seed_files
                        )
                    )


                    status.update(
                        label="SEED dataset loaded!",
                        state="complete",
                        expanded=False,
                    )


                    # ----------------------------------------
                    # RESULTS
                    # ----------------------------------------

                    st.markdown(
                        '<div class="section-title">Processing Complete</div>',
                        unsafe_allow_html=True,
                    )


                    c1, c2, c3, c4 = st.columns(
                        4
                    )


                    with c1:

                        st.metric(
                            "Samples",
                            f"{data_array.shape[0]:,}",
                        )


                    with c2:

                        st.metric(
                            "Frequency Bands",
                            data_array.shape[1],
                        )


                    with c3:

                        st.metric(
                            "EEG Channels",
                            data_array.shape[2],
                        )


                    with c4:

                        if subjects_array is not None:

                            subject_count = len(
                                np.unique(
                                    subjects_array
                                )
                            )

                        else:

                            subject_count = "N/A"


                        st.metric(
                            "Subjects",
                            subject_count,
                        )


                    st.success(
                        " SEED dataset is ready."
                    )


                    st.write(
                        "Processed representation:"
                    )


                    st.code(
                        str(
                            data_array.shape
                        )
                    )


                except Exception as error:

                    status.update(
                        label="SEED processing failed",
                        state="error",
                        expanded=True,
                    )

                    st.error(
                        f"SEED processing error: "
                        f"{error}"
                    )


# ============================================================
# ACTIVE DATASET
# ============================================================

if st.session_state.dataset_processed:

    active_dataset = (
        st.session_state.dataset_type
    )

    active_features = (
        st.session_state.processed_features
    )


    if active_features is not None:

        active_shape = (
            np.asarray(
                active_features
            ).shape
        )

    else:

        active_shape = "Unavailable"


    st.markdown(
        '<div class="section-title">Active Dataset</div>',
        unsafe_allow_html=True,
    )


    st.html(
        f"""
        <div class="success-card">

            <div class="success-title">
                 {active_dataset} dataset is loaded
            </div>

            <div class="success-text">

                Current representation:
                <b>{active_shape}</b>

                <br><br>

                The processed dataset is stored in the
                current Streamlit session.

                <br>

                You can now open
                <b>EEG Visualization</b>,
                <b>Feature Analysis</b>,
                and other pages without processing
                the dataset again.

            </div>

        </div>
        """
    )

