import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Evaluation",
    page_icon=None,
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

.eval-header {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #172554
    );
    color: white;
    margin-bottom: 28px;
}

.eval-title {
    font-size: 34px;
    font-weight: 800;
}

.eval-subtitle {
    color: #cbd5e1;
    margin-top: 8px;
    font-size: 15px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #0f172a;
    margin-top: 32px;
    margin-bottom: 16px;
}

.metric-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(15,23,42,0.05);
}

.metric-label {
    color: #64748b;
    font-size: 13px;
}

.metric-value {
    color: #0f172a;
    font-size: 28px;
    font-weight: 800;
    margin-top: 6px;
}

.best-card {
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-radius: 16px;
    padding: 24px;
}

.best-title {
    color: #1e3a8a;
    font-size: 20px;
    font-weight: 800;
}

.best-text {
    color: #334155;
    line-height: 1.7;
    margin-top: 10px;
}

.warning-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 16px;
    padding: 22px;
    color: #9a3412;
}

.success-card {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 22px;
    color: #166534;
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
<div class="eval-header">

<div class="eval-title">
Evaluation & Results
</div>

<div class="eval-subtitle">
Quantitative evaluation of the EEG emotion-recognition
models using Accuracy, Balanced Accuracy, Macro F1
and confusion matrices.
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# ACTUAL RESULTS
# ============================================================

# ------------------------------------------------------------
# DEAP
# ------------------------------------------------------------

deap_results = pd.DataFrame(
    [
        [
            "SVM",
            "Valence",
            0.7875,
            0.4961,
            0.4406,
        ],
        [
            "CNN",
            "Valence",
            0.7937,
            0.5000,
            0.4425,
        ],
        [
            "GCN + GRU",
            "Valence",
            0.7063,
            0.5234,
            0.5240,
        ],
        [
            "SVM",
            "Arousal",
            0.2938,
            0.4754,
            0.2904,
        ],
        [
            "CNN",
            "Arousal",
            0.7500,
            0.4839,
            0.4286,
        ],
        [
            "GCN + GRU",
            "Arousal",
            0.7750,
            0.5000,
            0.4366,
        ],
    ],
    columns=[
        "Model",
        "Task",
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
    ],
)


# ------------------------------------------------------------
# SEED
# ------------------------------------------------------------

seed_results = pd.DataFrame(
    [
        [
            "SVM",
            "-",
            0.3351,
            0.3403,
            0.2941,
        ],
        [
            "CNN",
            "-",
            0.3598,
            0.3665,
            0.2863,
        ],
        [
            "GCN",
            "-",
            0.4474,
            0.4450,
            0.4425,
        ],
        [
            "GCN + GRU",
            "20",
            0.4403,
            0.4330,
            0.3483,
        ],
        [
            "GCN + GRU",
            "10",
            0.4164,
            0.4149,
            0.4144,
        ],
    ],
    columns=[
        "Model",
        "Sequence Length",
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1",
    ],
)


# ============================================================
# CONVERT TO PERCENT
# ============================================================

deap_display = deap_results.copy()

seed_display = seed_results.copy()


for df in [
    deap_display,
    seed_display,
]:

    df["Accuracy"] = (
        df["Accuracy"] * 100
    ).round(2)

    df["Balanced Accuracy"] = (
        df["Balanced Accuracy"] * 100
    ).round(2)

    df["Macro F1"] = (
        df["Macro F1"] * 100
    ).round(2)


# ============================================================
# OVERALL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Evaluation Summary</div>',
    unsafe_allow_html=True,
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Datasets Evaluated</div>
<div class="metric-value">2</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c2:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">SEED Test Subjects</div>
<div class="metric-value">3</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c3:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Best SEED Accuracy</div>
<div class="metric-value">44.74%</div>
</div>
""",
        unsafe_allow_html=True,
    )


with c4:

    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">Best SEED Macro F1</div>
<div class="metric-value">44.25%</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# DEAP RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">DEAP Evaluation</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="success-card">

<b>Evaluation protocol:</b> subject-based train/validation/test
split. DEAP was evaluated separately for the two binary
emotion dimensions: Valence and Arousal.

</div>
""",
    unsafe_allow_html=True,
)


st.subheader("DEAP Model Comparison")


st.dataframe(
    deap_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DEAP TASK SELECTOR
# ============================================================

deap_task = st.radio(
    "Select DEAP Task",
    [
        "Valence",
        "Arousal",
    ],
    horizontal=True,
)


selected_deap = deap_display[
    deap_display["Task"] == deap_task
].copy()


# ============================================================
# DEAP ACCURACY CHART
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Bar(
        x=selected_deap["Model"],
        y=selected_deap["Accuracy"],
        name="Accuracy",
        text=[
            f"{x:.2f}%"
            for x in selected_deap["Accuracy"]
        ],
        textposition="aut✓,
    )
)


fig.add_trace(
    go.Bar(
        x=selected_deap["Model"],
        y=selected_deap["Balanced Accuracy"],
        name="Balanced Accuracy",
        text=[
            f"{x:.2f}%"
            for x in selected_deap[
                "Balanced Accuracy"
            ]
        ],
        textposition="aut✓,
    )
)


fig.add_trace(
    go.Bar(
        x=selected_deap["Model"],
        y=selected_deap["Macro F1"],
        name="Macro F1",
        text=[
            f"{x:.2f}%"
            for x in selected_deap["Macro F1"]
        ],
        textposition="aut✓,
    )
)


fig.update_layout(
    title=f"DEAP {deap_task} Model Performance",
    yaxis_title="Score (%)",
    yaxis_range=[0, 100],
    barmode="group",
    height=500,
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# DEAP BEST BY METRIC
# ============================================================

best_accuracy_row = selected_deap.loc[
    selected_deap["Accuracy"].idxmax()
]


best_balanced_row = selected_deap.loc[
    selected_deap[
        "Balanced Accuracy"
    ].idxmax()
]


best_f1_row = selected_deap.loc[
    selected_deap["Macro F1"].idxmax()
]


st.subheader(
    f"Best DEAP Results — {deap_task}"
)


b1, b2, b3 = st.columns(3)


with b1:

    st.metric(
        "Best Accuracy",
        f"{best_accuracy_row['Accuracy']:.2f}%",
        best_accuracy_row["Model"],
    )


with b2:

    st.metric(
        "Best Balanced Accuracy",
        f"{best_balanced_row['Balanced Accuracy']:.2f}%",
        best_balanced_row["Model"],
    )


with b3:

    st.metric(
        "Best Macro F1",
        f"{best_f1_row['Macro F1']:.2f}%",
        best_f1_row["Model"],
    )


# ============================================================
# DEAP CONFUSION MATRICES
# ============================================================

st.subheader(
    f"DEAP {deap_task} Confusion Matrices"
)


if deap_task == "Valence":

    deap_cm = {
        "SVM": np.array(
            [
                [126, 1],
                [33, 0],
            ]
        ),
        "CNN": np.array(
            [
                [127, 0],
                [33, 0],
            ]
        ),
        "GCN + GRU": np.array(
            [
                [106, 21],
                [26, 7],
            ]
        ),
    }

else:

    deap_cm = {
        "SVM": np.array(
            [
                [18, 106],
                [7, 29],
            ]
        ),
        "CNN": np.array(
            [
                [120, 4],
                [36, 0],
            ]
        ),
        "GCN + GRU": np.array(
            [
                [124, 0],
                [36, 0],
            ]
        ),
    }


cm_cols = st.columns(3)


class_names = [
    "Low",
    "High",
]


for col, (
    model_name,
    matrix,
) in zip(
    cm_cols,
    deap_cm.items(),
):

    with col:

        fig_cm = go.Figure(
            go.Heatmap(
                z=matrix,
                x=[
                    f"Pred {x}"
                    for x in class_names
                ],
                y=[
                    f"Actual {x}"
                    for x in class_names
                ],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )


        fig_cm.update_layout(
            title=model_name,
            height=350,
        )


        st.plotly_chart(
            fig_cm,
            use_container_width=True,
        )


# ============================================================
# SEED RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">SEED Evaluation</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="success-card">

<b>Evaluation protocol:</b> subject-based evaluation with
unseen subjects in the validation and test sets.

The SEED test set contains subjects not used during training.

</div>
""",
    unsafe_allow_html=True,
)


st.subheader("SEED Model Comparison")


st.dataframe(
    seed_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SEED PERFORMANCE CHART
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Bar(
        x=seed_display["Model"].astype(str)
        + " "
        + seed_display[
            "Sequence Length"
        ].astype(str),
        y=seed_display["Accuracy"],
        name="Accuracy",
        text=[
            f"{x:.2f}%"
            for x in seed_display["Accuracy"]
        ],
        textposition="aut✓,
    )
)


fig.add_trace(
    go.Bar(
        x=seed_display["Model"].astype(str)
        + " "
        + seed_display[
            "Sequence Length"
        ].astype(str),
        y=seed_display[
            "Balanced Accuracy"
        ],
        name="Balanced Accuracy",
        text=[
            f"{x:.2f}%"
            for x in seed_display[
                "Balanced Accuracy"
            ]
        ],
        textposition="aut✓,
    )
)


fig.add_trace(
    go.Bar(
        x=seed_display["Model"].astype(str)
        + " "
        + seed_display[
            "Sequence Length"
        ].astype(str),
        y=seed_display["Macro F1"],
        name="Macro F1",
        text=[
            f"{x:.2f}%"
            for x in seed_display[
                "Macro F1"
            ]
        ],
        textposition="aut✓,
    )
)


fig.update_layout(
    title="SEED Test Performance Comparison",
    yaxis_title="Score (%)",
    yaxis_range=[0, 100],
    barmode="group",
    height=520,
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# BEST SEED MODEL
# ============================================================

st.markdown(
    '<div class="section-title">Best Current SEED Model</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="best-card">

<div class="best-title">
GCN — Best Overall SEED Model
</div>

<div class="best-text">

<b>Test Accuracy:</b> 44.74%<br>

<b>Balanced Accuracy:</b> 44.50%<br>

<b>Macro F1:</b> 44.25%<br><br>

The GCN outperformed the SVM and CNN baselines on
the unseen-subject test split.

The GCN + GRU experiments were retained as temporal
experiments, but neither sequence configuration
outperformed the GCN.

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SEED CONFUSION MATRICES
# ============================================================

st.markdown(
    '<div class="section-title">SEED Confusion Matrices</div>',
    unsafe_allow_html=True,
)


seed_confusions = {

    "GCN": np.array(
        [
            [1448, 792, 1120],
            [1104, 1104, 1104],
            [1324, 183, 2003],
        ]
    ),

    "GCN + GRU (20)": np.array(
        [
            [6, 40, 116],
            [0, 52, 104],
            [0, 12, 156],
        ]
    ),

    "GCN + GRU (10)": np.array(
        [
            [142, 78, 110],
            [108, 108, 108],
            [140, 39, 166],
        ]
    ),
}


selected_seed_cm = st.selectbox(
    "Select SEED Model",
    list(seed_confusions.keys()),
)


matrix = seed_confusions[
    selected_seed_cm
]


fig_cm = go.Figure(
    go.Heatmap(
        z=matrix,
        x=[
            "Pred Negative",
            "Pred Neutral",
            "Pred Positive",
        ],
        y=[
            "Actual Negative",
            "Actual Neutral",
            "Actual Positive",
        ],
        text=matrix,
        texttemplate="%{text}",
        colorscale="Blues",
        colorbar_title="Samples",
    )
)


fig_cm.update_layout(
    title=f"{selected_seed_cm} — Test Confusion Matrix",
    height=500,
)


st.plotly_chart(
    fig_cm,
    use_container_width=True,
)


# ============================================================
# SEED CLASS PERFORMANCE
# ============================================================

st.subheader(
    f"{selected_seed_cm} Class Performance"
)


if selected_seed_cm == "GCN":

    class_df = pd.DataFrame(
        {
            "Emotion": [
                "Negative",
                "Neutral",
                "Positive",
            ],
            "Precision": [
                0.3736,
                0.5310,
                0.4739,
            ],
            "Recall": [
                0.4310,
                0.3333,
                0.5707,
            ],
            "F1": [
                0.4002,
                0.4096,
                0.5178,
            ],
        }
    )

elif selected_seed_cm == "GCN + GRU (20)":

    class_df = pd.DataFrame(
        {
            "Emotion": [
                "Negative",
                "Neutral",
                "Positive",
            ],
            "Precision": [
                1.0000,
                0.5000,
                0.4149,
            ],
            "Recall": [
                0.0370,
                0.3333,
                0.9286,
            ],
            "F1": [
                0.0714,
                0.4000,
                0.5735,
            ],
        }
    )

else:

    class_df = pd.DataFrame(
        {
            "Emotion": [
                "Negative",
                "Neutral",
                "Positive",
            ],
            "Precision": [
                0.3641,
                0.4800,
                0.4323,
            ],
            "Recall": [
                0.4303,
                0.3333,
                0.4812,
            ],
            "F1": [
                0.3944,
                0.3934,
                0.4554,
            ],
        }
    )


class_display = class_df.copy()

for column in [
    "Precision",
    "Recall",
    "F1",
]:

    class_display[column] = (
        class_display[column] * 100
    ).round(2)


st.dataframe(
    class_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# SEED CLASS PERFORMANCE CHART
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Bar(
        x=class_display["Emotion"],
        y=class_display["Precision"],
        name="Precision",
    )
)


fig.add_trace(
    go.Bar(
        x=class_display["Emotion"],
        y=class_display["Recall"],
        name="Recall",
    )
)


fig.add_trace(
    go.Bar(
        x=class_display["Emotion"],
        y=class_display["F1"],
        name="F1",
    )
)


fig.update_layout(
    title=f"{selected_seed_cm} — Class-wise Performance",
    yaxis_title="Score (%)",
    yaxis_range=[0, 100],
    barmode="group",
    height=450,
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# GENERALIZATION ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-title">Generalization Analysis</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="warning-card">

<b>Important observation:</b>

The SEED GCN model achieved a higher training performance
than its unseen-subject validation and test performance.

The GCN + GRU experiments showed an even stronger
train/validation gap. For example, the sequence-length-10
experiment reached approximately 76.79% training Macro F1,
while its best validation Macro F1 was 47.49%.

This indicates that the models have difficulty generalizing
to subjects that were not present during training.

Therefore, the test results should be interpreted as
evidence of the current model's generalization capability,
rather than as state-of-the-art performance.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# KEY FINDINGS
# ============================================================

st.markdown(
    '<div class="section-title">Key Findings</div>',
    unsafe_allow_html=True,
)


findings = [
    (
        "1",
        "Spatial graph learning helped",
        "The SEED GCN achieved 44.74% test accuracy and "
        "44.25% Macro F1, outperforming the current SVM "
        "and CNN baselines.",
    ),
    (
        "2",
        "Temporal modeling did not improve SEED performance",
        "Neither the 20-step nor 10-step GCN + GRU "
        "experiment outperformed the standalone GCN.",
    ),
    (
        "3",
        "Balanced metrics are important",
        "Accuracy alone can hide class-wise weaknesses. "
        "Balanced Accuracy and Macro F1 provide a more "
        "informative view of multi-class performance.",
    ),
    (
        "4",
        "Unseen-subject generalization remains difficult",
        "The difference between training and unseen-subject "
        "evaluation indicates substantial subject variability "
        "and model overfitting.",
    ),
]


for number, title, description in findings:

    st.markdown(
        f"""
<div class="metric-card"
style="text-align:left; margin-bottom:12px;">

<div class="metric-label">
Finding {number}
</div>

<div style="
font-size:18px;
font-weight:750;
color:#0f172a;
margin-top:5px;
">
{title}
</div>

<div style="
color:#475569;
margin-top:8px;
line-height:1.6;
">
{description}
</div>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# FINAL RESULT
# ============================================================

st.markdown(
    '<div class="section-title">Final Evaluation Statement</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="success-card">

The current experimental results demonstrate the complete
EEG emotion-recognition pipeline across DEAP and SEED,
including feature representation, spatial graph learning,
temporal modeling experiments and quantitative evaluation.

For the current SEED unseen-subject evaluation, the
standalone GCN is the strongest model among the tested
architectures, achieving:

<br><br>

<b>44.74% Accuracy</b> &nbsp; | &nbsp;
<b>44.50% Balanced Accuracy</b> &nbsp; | &nbsp;
<b>44.25% Macro F1</b>

<br><br>

The results also identify subject generalization and
temporal-model overfitting as important areas for future
improvement.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Evaluation • DEAP + SEED • Accuracy • Balanced Accuracy • Macro F1"
)