import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Architecture",
    page_icon=None,
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.arch-header {
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

.arch-title {
    font-size: 34px;
    font-weight: 800;
}

.arch-subtitle {
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

.arch-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 16px;
    padding: 22px;
    height: 100%;
    box-shadow: 0 4px 15px rgba(15,23,42,0.06);
}

.arch-card h3 {
    color: #0f172a;
    margin-bottom: 8px;
}

.arch-card p {
    color: #475569;
    line-height: 1.6;
}

.metric-card {
    background: white;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
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

.flow-box {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    min-height: 90px;
}

.flow-number {
    color: #2563eb;
    font-size: 14px;
    font-weight: 800;
}

.flow-title {
    color: #0f172a;
    font-size: 17px;
    font-weight: 750;
    margin-top: 5px;
}

.flow-detail {
    color: #64748b;
    font-size: 12px;
    margin-top: 5px;
}

.arrow {
    text-align: center;
    font-size: 25px;
    color: #2563eb;
    padding-top: 25px;
}

.info-panel {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    padding: 20px;
    color: #1e3a8a;
}

.warning-panel {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    border-radius: 14px;
    padding: 20px;
    color: #9a3412;
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
<div class="arch-header">

<div class="arch-title">
Model Architecture
</div>

<div class="arch-subtitle">
Spatial-temporal representation learning architecture
for EEG-based emotion recognition
</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Architecture Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="info-panel">

The project combines EEG preprocessing and feature
representation with spatial graph learning and temporal
sequence modeling.

The main research idea is to preserve the spatial
relationships between EEG electrodes instead of treating
all EEG channels as independent features.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">DEAP EEG Channels</div>
<div class="metric-value">32</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">SEED EEG Channels</div>
<div class="metric-value">62</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">SEED GCN Parameters</div>
<div class="metric-value">4,483</div>
</div>
""",
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        """
<div class="metric-card">
<div class="metric-label">DEAP GCN + GRU Parameters</div>
<div class="metric-value">46,690</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

st.markdown(
    '<div class="section-title">Complete Processing Pipeline</div>',
    unsafe_allow_html=True,
)


steps = [
    (
        "01",
        "EEG Data",
        "DEAP / SEED",
    ),
    (
        "02",
        "Preprocessing",
        "Cleaning and normalization",
    ),
    (
        "03",
        "Feature Representation",
        "DE + PSD",
    ),
    (
        "04",
        "Spatial Learning",
        "Electrode graph + GCN",
    ),
    (
        "05",
        "Temporal Learning",
        "GRU",
    ),
    (
        "06",
        "Classification",
        "Emotion output",
    ),
]


for i in range(0, len(steps), 3):

    cols = st.columns(3)

    for j, col in enumerate(cols):

        index = i + j

        if index >= len(steps):
            continue

        number, title, detail = steps[index]

        with col:

            st.markdown(
                f"""
<div class="flow-box">

<div class="flow-number">{number}</div>

<div class="flow-title">
{title}
</div>

<div class="flow-detail">
{detail}
</div>

</div>
""",
                unsafe_allow_html=True,
            )

    if i < len(steps) - 3:

        st.markdown(
            '<div class="arrow"></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# DATA REPRESENTATION
# ============================================================

st.markdown(
    '<div class="section-title">Dataset Representation</div>',
    unsafe_allow_html=True,
)


col1, col2 = st.columns(2)


with col1:

    st.markdown(
        """
<div class="arch-card">

<h3>DEAP Representation</h3>

<p>
The processed DEAP representation contains:
</p>

<ul>
<li>32 EEG channels</li>
<li>30 temporal windows per trial</li>
<li>5 Differential Entropy features</li>
<li>5 Power Spectral Density features</li>
<li>10 features per EEG channel</li>
</ul>

<p>
Final representation:
<b>30  32  10</b>
</p>

<p>
The DEAP task predicts two binary emotion
dimensions: <b>Valence</b> and <b>Arousal</b>.
</p>

</div>
""",
        unsafe_allow_html=True,
    )


with col2:

    st.markdown(
        """
<div class="arch-card">

<h3>SEED Representation</h3>

<p>
The processed SEED representation contains:
</p>

<ul>
<li>62 EEG electrodes</li>
<li>5 frequency-band features</li>
<li>Negative, Neutral and Positive classes</li>
<li>Electrode graph representation</li>
</ul>

<p>
Final sample representation:
<b>5  62</b>
</p>

<p>
The 62 electrodes are represented as nodes
in the spatial EEG graph.
</p>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# GCN EXPLANATION
# ============================================================

st.markdown(
    '<div class="section-title">Spatial Learning: Graph Convolutional Network</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="info-panel">

<b>Why GCN</b><br><br>

EEG electrodes are located at different positions on the
scalp. A graph representation allows the model to represent
relationships between spatially related electrodes.

Each EEG electrode becomes a graph node, while edges
represent spatial relationships between electrodes.

The GCN then propagates information across the graph to
learn spatial EEG representations.

</div>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# GCN FLOW
# ------------------------------------------------------------

g1, g2, g3, g4 = st.columns(4)


with g1:
    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">INPUT</div>
<div class="flow-title">EEG Nodes</div>
<div class="flow-detail">62 electrodes for SEED</div>
</div>
""",
        unsafe_allow_html=True,
    )


with g2:
    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">GCN LAYER 1</div>
<div class="flow-title">5  32</div>
<div class="flow-detail">Spatial feature extraction</div>
</div>
""",
        unsafe_allow_html=True,
    )


with g3:
    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">GCN LAYER 2</div>
<div class="flow-title">32  64</div>
<div class="flow-detail">Higher-level spatial features</div>
</div>
""",
        unsafe_allow_html=True,
    )


with g4:
    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">POOLING</div>
<div class="flow-title">Mean Pooling</div>
<div class="flow-detail">62 nodes  graph representation</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SEED GCN ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section-title">SEED GCN Architecture</div>',
    unsafe_allow_html=True,
)


seed_layers = [
    "Input: 5 features  62 electrodes",
    "GCNConv: 5  32",
    "ReLU + Dropout",
    "GCNConv: 32  64",
    "ReLU + Dropout",
    "Mean pooling across 62 nodes",
    "Linear: 64  32",
    "ReLU + Dropout",
    "Linear: 32  3",
    "Output: Negative / Neutral / Positive",
]


fig = go.Figure()


for i, layer in enumerate(seed_layers):

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[-i],
            mode="markers+text",
            marker=dict(
                size=30,
            ),
            text=[layer],
            textposition="middle right",
            showlegend=False,
        )
    )


fig.update_layout(
    height=650,
    title="SEED GCN Layer Flow",
    xaxis=dict(
        visible=False,
        range=[-1, 8],
    ),
    yaxis=dict(
        visible=False,
    ),
    margin=dict(
        l=20,
        r=20,
        t=70,
        b=20,
    ),
)


st.plotly_chart(
    fig,
    use_container_width=True,
)


# ============================================================
# TEMPORAL LEARNING
# ============================================================

st.markdown(
    '<div class="section-title">Temporal Learning: GRU</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="arch-card">

<h3>Why GRU</h3>

<p>
EEG signals contain temporal dependencies. The GRU is used
to model how the learned spatial representation changes
over a sequence of EEG observations.
</p>

<p>
The spatial representation generated by the GCN becomes
the input sequence for the GRU.
</p>

</div>
""",
    unsafe_allow_html=True,
)


t1, t2, t3 = st.columns(3)


with t1:

    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">STEP 01</div>
<div class="flow-title">GCN Features</div>
<div class="flow-detail">
Spatial representation for each observation
</div>
</div>
""",
        unsafe_allow_html=True,
    )


with t2:

    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">STEP 02</div>
<div class="flow-title">GRU</div>
<div class="flow-detail">
Learns temporal dependencies
</div>
</div>
""",
        unsafe_allow_html=True,
    )


with t3:

    st.markdown(
        """
<div class="flow-box">
<div class="flow-number">STEP 03</div>
<div class="flow-title">Classifier</div>
<div class="flow-detail">
Produces emotion prediction
</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# DEAP GCN + GRU
# ============================================================

st.markdown(
    '<div class="section-title">DEAP Spatial-Temporal Model</div>',
    unsafe_allow_html=True,
)


d1, d2 = st.columns(2)


with d1:

    st.markdown(
        """
<div class="arch-card">

<h3>DEAP Input</h3>

<p>
Each processed trial is represented as:
</p>

<p style="font-size:24px; font-weight:800;">
30  32  10
</p>

<p>
30 temporal windows<br>
32 EEG channels<br>
10 features per channel
</p>

<p>
The 10 features combine Differential Entropy
and Power Spectral Density representations.
</p>

</div>
""",
        unsafe_allow_html=True,
    )


with d2:

    st.markdown(
        """
<div class="arch-card">

<h3>DEAP GCN + GRU</h3>

<p>
The spatial component learns relationships between
EEG channels, while the GRU models temporal dependencies.
</p>

<p>
<b>Parameters:</b> 46,690
</p>

<p>
The model was evaluated separately for:
</p>

<ul>
<li>Valence</li>
<li>Arousal</li>
</ul>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SEED GCN + GRU
# ============================================================

st.markdown(
    '<div class="section-title">SEED Spatial-Temporal Experiment</div>',
    unsafe_allow_html=True,
)


s1, s2 = st.columns(2)


with s1:

    st.markdown(
        """
<div class="arch-card">

<h3>SEED GCN</h3>

<p>
Architecture:
</p>

<ul>
<li>GCNConv 5  32</li>
<li>GCNConv 32  64</li>
<li>Mean graph pooling</li>
<li>Linear 64  32</li>
<li>Linear 32  3</li>
</ul>

<p>
<b>Parameters: 4,483</b>
</p>

</div>
""",
        unsafe_allow_html=True,
    )


with s2:

    st.markdown(
        """
<div class="arch-card">

<h3>SEED GCN + GRU</h3>

<p>
The temporal experiment added a GRU after
the spatial GCN representation.
</p>

<ul>
<li>GCN spatial representation</li>
<li>GRU hidden size: 64</li>
<li>Classifier: 64  32  3</li>
</ul>

<p>
<b>Parameters: 29,443</b>
</p>

</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">SEED Model Architecture Comparison</div>',
    unsafe_allow_html=True,
)


comparison = {
    "Model": [
        "SVM",
        "CNN",
        "GCN",
        "GCN + GRU (20)",
        "GCN + GRU (10)",
    ],
    "Representation": [
        "Flattened features",
        "2D feature representation",
        "EEG electrode graph",
        "Graph + temporal sequence",
        "Graph + temporal sequence",
    ],
    "Test Accuracy": [
        33.51,
        35.98,
        44.74,
        44.03,
        41.64,
    ],
    "Balanced Accuracy": [
        34.03,
        36.65,
        44.50,
        43.30,
        41.49,
    ],
    "Macro F1": [
        29.41,
        28.63,
        44.25,
        34.83,
        41.44,
    ],
}


import pandas as pd

comparison_df = pd.DataFrame(
    comparison
)


st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# VISUAL MODEL COMPARISON
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Bar(
        name="Accuracy",
        x=comparison_df["Model"],
        y=comparison_df["Test Accuracy"],
    )
)


fig.add_trace(
    go.Bar(
        name="Balanced Accuracy",
        x=comparison_df["Model"],
        y=comparison_df["Balanced Accuracy"],
    )
)


fig.add_trace(
    go.Bar(
        name="Macro F1",
        x=comparison_df["Model"],
        y=comparison_df["Macro F1"],
    )
)


fig.update_layout(
    title="SEED Test Performance by Architecture",
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
# BEST MODEL
# ============================================================

st.markdown(
    '<div class="section-title">Best Current SEED Architecture</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="info-panel">

<b>GCN is currently the best-performing SEED model in this
project evaluation.</b>

<br><br>

Test Accuracy: <b>44.74%</b><br>
Balanced Accuracy: <b>44.50%</b><br>
Macro F1: <b>44.25%</b>

<br><br>

The GCN outperformed the SVM and CNN baselines on the
unseen-subject test split.

The GCN + GRU experiments were retained as temporal
experiments, but did not outperform the GCN on the current
SEED split.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">Research Interpretation</div>',
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="warning-panel">

<b>Important:</b> The reported results are test-set results
from the current subject-based evaluation and should not be
interpreted as state-of-the-art performance.

The main architectural finding from the current experiments
is that representing EEG electrodes as a spatial graph
provided a stronger SEED representation than the current
SVM and CNN baselines.

The GCN + GRU experiments demonstrate temporal modeling,
but the current results show that adding the GRU did not
improve generalization on the unseen SEED subjects.

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FINAL PIPELINE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Final Research Pipeline</div>',
    unsafe_allow_html=True,
)


final_cols = st.columns(6)


final_steps = [
    ("01", "EEG", "DEAP / SEED"),
    ("02", "Features", "DE + PSD"),
    ("03", "Graph", "Electrode topology"),
    ("04", "GCN", "Spatial learning"),
    ("05", "GRU", "Temporal learning"),
    ("06", "Output", "Emotion"),
]


for col, (num, title, detail) in zip(
    final_cols,
    final_steps
):

    with col:

        st.markdown(
            f"""
<div class="flow-box">

<div class="flow-number">
{num}
</div>

<div class="flow-title">
{title}
</div>

<div class="flow-detail">
{detail}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Model Architecture  EEG Emotion Recognition  DEAP + SEED"
)