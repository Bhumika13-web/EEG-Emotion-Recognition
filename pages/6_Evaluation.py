import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

DEAP_RESULTS_DIR = RESULTS_DIR / "DEAP"
SEED_RESULTS_DIR = RESULTS_DIR / "SEED"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Model Evaluation")

st.caption(
    "Evaluation of the EEG emotion recognition models using "
    "Accuracy, Balanced Accuracy, Macro F1-score, model "
    "comparison and performance visualizations."
)


# ============================================================
# DATASET SELECTION
# ============================================================

dataset_type = st.session_state.get(
    "dataset_type",
    None,
)


if dataset_type is None:

    st.warning(
        "No dataset has been selected."
    )

    st.info(
        """
        Please go to **Upload Dataset** first and select
        either **DEAP** or **SEED**.

        The Evaluation page automatically displays the
        evaluation results corresponding to the selected dataset.
        """
    )

    st.stop()


dataset_type = str(
    dataset_type
).strip().upper()


if dataset_type not in ["DEAP", "SEED"]:

    st.error(
        f"Unsupported dataset type: {dataset_type}"
    )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(
    dataframe,
    possible_names,
):
    """
    Find the first matching column.
    """

    if dataframe is None:
        return None

    for name in possible_names:

        if name in dataframe.columns:
            return name

    return None


def load_csv(
    path,
):
    """
    Load CSV safely.
    """

    if not path.exists():

        return None

    try:

        return pd.read_csv(path)

    except Exception as error:

        st.error(
            f"Unable to read {path.name}: {error}"
        )

        return None


def to_numeric(
    value,
):
    """
    Convert a value to numeric.
    """

    try:

        return float(value)

    except Exception:

        return None


def percentage(
    value,
):
    """
    Convert decimal metric to percentage.
    """

    numeric_value = to_numeric(
        value
    )

    if numeric_value is None:

        return "N/A"

    return f"{numeric_value * 100:.2f}%"


def find_best_model(
    dataframe,
):
    """
    Select best model using Macro F1 first,
    then Balanced Accuracy, then Accuracy.
    """

    if dataframe is None:
        return None

    if dataframe.empty:
        return None


    macro_f1_column = find_column(
        dataframe,
        [
            "Macro F1",
            "Macro_F1",
            "macro_f1",
            "macro_f1_score",
        ],
    )


    if macro_f1_column is not None:

        values = pd.to_numeric(
            dataframe[
                macro_f1_column
            ],
            errors="coerce",
        )

        if values.notna().any():

            return dataframe.loc[
                values.idxmax()
            ]


    balanced_column = find_column(
        dataframe,
        [
            "Balanced Accuracy",
            "Balanced_Accuracy",
            "balanced_accuracy",
        ],
    )


    if balanced_column is not None:

        values = pd.to_numeric(
            dataframe[
                balanced_column
            ],
            errors="coerce",
        )

        if values.notna().any():

            return dataframe.loc[
                values.idxmax()
            ]


    accuracy_column = find_column(
        dataframe,
        [
            "Accuracy",
            "accuracy",
        ],
    )


    if accuracy_column is not None:

        values = pd.to_numeric(
            dataframe[
                accuracy_column
            ],
            errors="coerce",
        )

        if values.notna().any():

            return dataframe.loc[
                values.idxmax()
            ]


    return dataframe.iloc[0]


def show_metrics(
    row,
    accuracy_column,
    balanced_column,
    macro_f1_column,
):
    """
    Display three main evaluation metrics.
    """

    accuracy = (
        row[accuracy_column]
        if accuracy_column is not None
        else None
    )

    balanced = (
        row[balanced_column]
        if balanced_column is not None
        else None
    )

    macro_f1 = (
        row[macro_f1_column]
        if macro_f1_column is not None
        else None
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Accuracy",
            percentage(accuracy),
        )


    with col2:

        st.metric(
            "Balanced Accuracy",
            percentage(balanced),
        )


    with col3:

        st.metric(
            "Macro F1",
            percentage(macro_f1),
        )


def create_comparison_chart(
    dataframe,
    model_column,
    accuracy_column,
    balanced_column,
    macro_f1_column,
    title,
):
    """
    Create interactive model comparison chart.
    """

    if dataframe is None:
        return

    if dataframe.empty:
        return

    if model_column is None:
        return


    models = dataframe[
        model_column
    ].astype(str)


    fig = go.Figure()


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if accuracy_column is not None:

        values = pd.to_numeric(
            dataframe[
                accuracy_column
            ],
            errors="coerce",
        ).fillna(0)


        fig.add_trace(
            go.Bar(
                x=models,
                y=values * 100,
                name="Accuracy",
                text=[
                    f"{v * 100:.2f}%"
                    for v in values
                ],
                textposition="auto",
            )
        )


    # --------------------------------------------------------
    # BALANCED ACCURACY
    # --------------------------------------------------------

    if balanced_column is not None:

        values = pd.to_numeric(
            dataframe[
                balanced_column
            ],
            errors="coerce",
        ).fillna(0)


        fig.add_trace(
            go.Bar(
                x=models,
                y=values * 100,
                name="Balanced Accuracy",
                text=[
                    f"{v * 100:.2f}%"
                    for v in values
                ],
                textposition="auto",
            )
        )


    # --------------------------------------------------------
    # MACRO F1
    # --------------------------------------------------------

    if macro_f1_column is not None:

        values = pd.to_numeric(
            dataframe[
                macro_f1_column
            ],
            errors="coerce",
        ).fillna(0)


        fig.add_trace(
            go.Bar(
                x=models,
                y=values * 100,
                name="Macro F1",
                text=[
                    f"{v * 100:.2f}%"
                    for v in values
                ],
                textposition="auto",
            )
        )


    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode="group",
        height=500,
        margin=dict(
            l=50,
            r=30,
            t=80,
            b=80,
        ),
        legend_title="Metric",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def show_image(
    path,
    title,
):
    """
    Display an image if it exists.
    """

    if path.exists():

        st.markdown(
            f"**{title}**"
        )

        st.image(
            str(path),
            use_container_width=True,
        )

    else:

        st.warning(
            f"File not found: {path.name}"
        )


# ============================================================
# ============================================================
# DEAP EVALUATION
# ============================================================
# ============================================================

if dataset_type == "DEAP":

    st.success(
        "DEAP dataset selected — showing DEAP evaluation only."
    )


    # ========================================================
    # DEAP INTRODUCTION
    # ========================================================

    st.header(
        "DEAP Evaluation"
    )

    st.write(
        """
        The DEAP experiment evaluates binary emotion
        classification for **Valence** and **Arousal**.

        The evaluation compares SVM, CNN and GCN + GRU
        models using Accuracy, Balanced Accuracy and
        Macro F1-score.
        """
    )


    st.divider()


    # ========================================================
    # LOAD DEAP MODEL COMPARISON
    # ========================================================

    deap_csv_path = (
        DEAP_RESULTS_DIR
        / "deap_model_comparison.csv"
    )


    deap_df = load_csv(
        deap_csv_path
    )


    if deap_df is None:

        st.error(
            "DEAP model comparison file was not found."
        )

        st.code(
            "results/DEAP/deap_model_comparison.csv"
        )

        st.stop()


    # ========================================================
    # DEAP COLUMNS
    # ========================================================

    deap_model_column = find_column(
        deap_df,
        [
            "Model",
            "model",
            "Model Name",
            "model_name",
        ],
    )


    deap_task_column = find_column(
        deap_df,
        [
            "Task",
            "task",
            "Target",
        ],
    )


    deap_accuracy_column = find_column(
        deap_df,
        [
            "Accuracy",
            "accuracy",
        ],
    )


    deap_balanced_column = find_column(
        deap_df,
        [
            "Balanced Accuracy",
            "Balanced_Accuracy",
            "balanced_accuracy",
        ],
    )


    deap_macro_f1_column = find_column(
        deap_df,
        [
            "Macro F1",
            "Macro_F1",
            "macro_f1",
            "macro_f1_score",
        ],
    )


    # ========================================================
    # DEAP MODEL COMPARISON
    # ========================================================

    st.subheader(
        "DEAP Model Comparison"
    )


    st.dataframe(
        deap_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # DEAP TASK SELECTION
    # ========================================================

    task_df = deap_df.copy()

    selected_task = "All"


    if deap_task_column is not None:

        task_values = (
            deap_df[
                deap_task_column
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )


        task_options = [
            "All"
        ] + task_values


        st.subheader(
            "DEAP Task"
        )


        selected_task = st.selectbox(
            "Select DEAP evaluation task",
            task_options,
            key="deap_evaluation_task",
        )


        if selected_task != "All":

            task_df = deap_df[
                deap_df[
                    deap_task_column
                ].astype(str)
                == selected_task
            ].copy()


    # ========================================================
    # BEST DEAP MODEL
    # ========================================================

    best_deap_row = find_best_model(
        task_df
    )


    if best_deap_row is not None:

        st.subheader(
            "Best DEAP Model"
        )


        best_model = (
            str(
                best_deap_row[
                    deap_model_column
                ]
            )
            if deap_model_column is not None
            else "Best Model"
        )


        st.success(
            f"Best Model: {best_model}"
        )


        show_metrics(
            best_deap_row,
            deap_accuracy_column,
            deap_balanced_column,
            deap_macro_f1_column,
        )


    # ========================================================
    # DEAP INTERACTIVE GRAPH
    # ========================================================

    st.subheader(
        "DEAP Model Performance"
    )


    chart_title = (
        "DEAP Model Comparison"
        if selected_task == "All"
        else f"DEAP Model Comparison - {selected_task}"
    )


    create_comparison_chart(
        dataframe=task_df,
        model_column=deap_model_column,
        accuracy_column=deap_accuracy_column,
        balanced_column=deap_balanced_column,
        macro_f1_column=deap_macro_f1_column,
        title=chart_title,
    )


    # ========================================================
    # DEAP SAVED GRAPHS
    # ========================================================

    st.subheader(
        "DEAP Evaluation Graphs"
    )


    if selected_task == "Valence":

        col1, col2 = st.columns(2)

        with col1:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_valence_accuracy.png",
                "Valence Accuracy",
            )

        with col2:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_valence_balanced_accuracy.png",
                "Valence Balanced Accuracy",
            )


        show_image(
            DEAP_RESULTS_DIR
            / "deap_valence_macro_f1.png",
            "Valence Macro F1",
        )


    elif selected_task == "Arousal":

        col1, col2 = st.columns(2)

        with col1:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_arousal_accuracy.png",
                "Arousal Accuracy",
            )

        with col2:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_arousal_balanced_accuracy.png",
                "Arousal Balanced Accuracy",
            )


        show_image(
            DEAP_RESULTS_DIR
            / "deap_arousal_macro_f1.png",
            "Arousal Macro F1",
        )


    else:

        st.markdown(
            "### Valence"
        )


        col1, col2 = st.columns(2)

        with col1:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_valence_accuracy.png",
                "Valence Accuracy",
            )

        with col2:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_valence_balanced_accuracy.png",
                "Valence Balanced Accuracy",
            )


        show_image(
            DEAP_RESULTS_DIR
            / "deap_valence_macro_f1.png",
            "Valence Macro F1",
        )


        st.markdown(
            "### Arousal"
        )


        col1, col2 = st.columns(2)

        with col1:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_arousal_accuracy.png",
                "Arousal Accuracy",
            )

        with col2:

            show_image(
                DEAP_RESULTS_DIR
                / "deap_arousal_balanced_accuracy.png",
                "Arousal Balanced Accuracy",
            )


        show_image(
            DEAP_RESULTS_DIR
            / "deap_arousal_macro_f1.png",
            "Arousal Macro F1",
        )


    # ========================================================
    # DEAP INTERPRETATION
    # ========================================================

    st.subheader(
        "DEAP Evaluation Interpretation"
    )


    st.info(
        """
        The DEAP experiment performs binary Valence and
        Arousal classification.

        Accuracy is considered together with Balanced
        Accuracy and Macro F1 because class imbalance can
        make raw accuracy misleading.

        SVM and CNN are used as baseline models, while
        GCN + GRU represents the spatial-temporal learning
        architecture.

        GCN models relationships between EEG electrodes,
        while GRU models temporal dependencies across
        EEG windows.
        """
    )


# ============================================================
# ============================================================
# SEED EVALUATION
# ============================================================
# ============================================================

elif dataset_type == "SEED":

    st.success(
        "SEED dataset selected — showing SEED evaluation only."
    )


    # ========================================================
    # SEED INTRODUCTION
    # ========================================================

    st.header(
        "SEED Evaluation"
    )


    st.write(
        """
        The SEED experiment performs three-class emotion
        recognition using Negative, Neutral and Positive
        emotion classes.

        The evaluation compares SVM, CNN, GCN and GCN + GRU
        models.
        """
    )


    st.divider()


    # ========================================================
    # LOAD SEED MODEL COMPARISON
    # ========================================================

    seed_csv_path = (
        SEED_RESULTS_DIR
        / "seed_model_comparison.csv"
    )


    seed_df = load_csv(
        seed_csv_path
    )


    if seed_df is None:

        st.error(
            "SEED model comparison file was not found."
        )

        st.code(
            "results/SEED/seed_model_comparison.csv"
        )

        st.stop()


    # ========================================================
    # SEED COLUMNS
    # ========================================================

    seed_model_column = find_column(
        seed_df,
        [
            "Model",
            "model",
            "Model Name",
            "model_name",
        ],
    )


    seed_accuracy_column = find_column(
        seed_df,
        [
            "Accuracy",
            "accuracy",
        ],
    )


    seed_balanced_column = find_column(
        seed_df,
        [
            "Balanced Accuracy",
            "Balanced_Accuracy",
            "balanced_accuracy",
        ],
    )


    seed_macro_f1_column = find_column(
        seed_df,
        [
            "Macro F1",
            "Macro_F1",
            "macro_f1",
            "macro_f1_score",
        ],
    )


    # ========================================================
    # SEED MODEL COMPARISON
    # ========================================================

    st.subheader(
        "SEED Model Comparison"
    )


    st.dataframe(
        seed_df,
        use_container_width=True,
        hide_index=True,
    )


    # ========================================================
    # BEST SEED MODEL
    # ========================================================

    best_seed_row = find_best_model(
        seed_df
    )


    if best_seed_row is not None:

        st.subheader(
            "Best SEED Model"
        )


        best_model = (
            str(
                best_seed_row[
                    seed_model_column
                ]
            )
            if seed_model_column is not None
            else "Best Model"
        )


        st.success(
            f"Best Model: {best_model}"
        )


        show_metrics(
            best_seed_row,
            seed_accuracy_column,
            seed_balanced_column,
            seed_macro_f1_column,
        )


    # ========================================================
    # SEED INTERACTIVE GRAPH
    # ========================================================

    st.subheader(
        "SEED Model Performance"
    )


    create_comparison_chart(
        dataframe=seed_df,
        model_column=seed_model_column,
        accuracy_column=seed_accuracy_column,
        balanced_column=seed_balanced_column,
        macro_f1_column=seed_macro_f1_column,
        title="SEED Model Comparison",
    )


    # ========================================================
    # SEED SAVED GRAPHS
    # ========================================================

    st.subheader(
        "SEED Evaluation Graphs"
    )


    col1, col2 = st.columns(2)


    with col1:

        show_image(
            SEED_RESULTS_DIR
            / "seed_accuracy_comparison.png",
            "Accuracy Comparison",
        )


    with col2:

        show_image(
            SEED_RESULTS_DIR
            / "seed_balanced_accuracy_comparison.png",
            "Balanced Accuracy Comparison",
        )


    show_image(
        SEED_RESULTS_DIR
        / "seed_macro_f1_comparison.png",
        "Macro F1 Comparison",
    )


    # ========================================================
    # SEED CONFUSION MATRICES
    # ========================================================

    st.subheader(
        "SEED Confusion Matrices"
    )


    st.write(
        "Confusion matrices for the trained GCN and "
        "GCN + GRU models."
    )


    col1, col2 = st.columns(2)


    with col1:

        show_image(
            SEED_RESULTS_DIR
            / "seed_gcn_confusion_matrix.png",
            "GCN Confusion Matrix",
        )


    with col2:

        show_image(
            SEED_RESULTS_DIR
            / "seed_gcn_gru10_confusion_matrix.png",
            "GCN + GRU — Sequence Length 10",
        )


    show_image(
        SEED_RESULTS_DIR
        / "seed_gcn_gru20_confusion_matrix.png",
        "GCN + GRU — Sequence Length 20",
    )


    # ========================================================
    # SEED EMOTION CLASSES
    # ========================================================

    st.subheader(
        "SEED Emotion Classes"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Class 0",
            "Negative",
        )


    with col2:

        st.metric(
            "Class 1",
            "Neutral",
        )


    with col3:

        st.metric(
            "Class 2",
            "Positive",
        )


    # ========================================================
    # SEED INTERPRETATION
    # ========================================================

    st.subheader(
        "SEED Evaluation Interpretation"
    )


    st.info(
        """
        The SEED experiment performs three-class emotion
        recognition using Negative, Neutral and Positive
        classes.

        SVM and CNN are used as baseline models.

        GCN models spatial relationships between the
        62 EEG electrodes.

        GCN + GRU additionally models temporal dependencies
        across EEG sequences.

        Accuracy, Balanced Accuracy and Macro F1 are
        considered together for evaluating generalization
        to unseen subjects.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EEG based Emotion Recognition using "
    "Spatial-Temporal Representation Learning "
    "· Dataset-specific evaluation"
)