import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding: 2rem 2rem 2rem 2rem; }
    h1 { color: #ffffff; font-size: 2rem; font-weight: 700; }
    h2 { color: #e0e0e0; font-size: 1.3rem; font-weight: 600; }
    h3 { color: #c0c0c0; font-size: 1.1rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #4e8cff;
        margin-bottom: 0.5rem;
    }
    .metric-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ffffff; font-size: 1.8rem; font-weight: 700; }
    .metric-sub   { color: #aaa; font-size: 0.85rem; margin-top: 2px; }
    .section-header {
        background: #1e2130;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 1.5rem 0 1rem 0;
        color: #4e8cff;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .anomaly-badge {
        display: inline-block;
        background: #ff4b4b22;
        color: #ff4b4b;
        border: 1px solid #ff4b4b55;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .robust-badge {
        display: inline-block;
        background: #ffa50022;
        color: #ffa500;
        border: 1px solid #ffa50055;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #4e8cff;
    }
    div[data-testid="stMetric"] label { color: #888 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #fff !important; }
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_data
def load_data():
    rf  = pd.read_csv(os.path.join(BASE, "results_random_forest.csv"))
    dt  = pd.read_csv(os.path.join(BASE, "results_decision_tree.csv"))
    lr  = pd.read_csv(os.path.join(BASE, "results_linear.csv"))
    svr = pd.read_csv(os.path.join(BASE, "results_svr.csv"))
    rf["Model"]  = "Random Forest"
    dt["Model"]  = "Decision Tree"
    lr["Model"]  = "Linear Regression"
    svr["Model"] = "SVR"
    return pd.concat([rf, dt, lr, svr], ignore_index=True), rf, dt, lr, svr

all_df, rf_df, dt_df, lr_df, svr_df = load_data()

METRICS = {
    "Random Forest":     {"R2": 0.9198, "RMSE": 0.1201, "MAE": 0.0850},
    "Decision Tree":     {"R2": 0.8324, "RMSE": 0.1735, "MAE": 0.1243},
    "Linear Regression": {"R2": 0.9365, "RMSE": 0.1068, "MAE": 0.0760},
    "SVR":               {"R2": 0.7501, "RMSE": 0.2119, "MAE": 0.1247},
}

SVR_IDX = [765,1557,1641,1771,727,742,1014,790,1536,432,1321,1766,1027]
RF_IDX  = [423,1537,436,44,1771,2095,433,432,1766,2096,456]
LR_IDX  = [433,2233,1356,2114,2567,2226,1954,108,2727,790,2323,942,2858,1338,695,767,210,1220,742,1289]
DT_IDX  = [2858,695,2658,940,2044,790,1289]

MODEL_COLORS = {
    "Random Forest":     "#4e8cff",
    "Decision Tree":     "#ff6b6b",
    "Linear Regression": "#2ecc71",
    "SVR":               "#f39c12",
}

DFRAME_MAP = {"Random Forest": rf_df, "Decision Tree": dt_df, "Linear Regression": lr_df, "SVR": svr_df}
IDX_MAP    = {"Random Forest": RF_IDX, "Decision Tree": DT_IDX, "Linear Regression": LR_IDX, "SVR": SVR_IDX}

# ── Plotly theming helpers ────────────────────────────────────────────────────
def style_fig(fig, title=None, height=440, showlegend=True):
    """Apply one consistent dark theme to every chart in the app."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=15)) if title else None,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e2130",
        font=dict(color="#dddddd", family="Arial, sans-serif"),
        legend=dict(bgcolor="#1e2130", bordercolor="#333333", borderwidth=1, font=dict(color="white")),
        showlegend=showlegend,
        margin=dict(l=50, r=30, t=55 if title else 25, b=45),
        hoverlabel=dict(bgcolor="#1e2130", font_color="white", bordercolor="#4e8cff"),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2a2d3a", zeroline=False, color="#cccccc", linecolor="#333333")
    fig.update_yaxes(showgrid=True, gridcolor="#2a2d3a", zeroline=False, color="#cccccc", linecolor="#333333")
    return fig

def hover_for(df):
    """customdata + hovertemplate so every house-level scatter shows full detail on hover."""
    customdata = np.stack([df["House_Index"], df["Actual"], df["Predicted"], df["Residual"]], axis=-1)
    template = ("House #%{customdata[0]}<br>"
                "Actual: $%{customdata[1]:,.0f}<br>"
                "Predicted: $%{customdata[2]:,.0f}<br>"
                "Residual: $%{customdata[3]:,.0f}<extra></extra>")
    return customdata, template

# ── Chart builders ────────────────────────────────────────────────────────────
def fig_metric_bars():
    models = list(METRICS.keys())
    fig = make_subplots(rows=1, cols=3, subplot_titles=("R\u00b2 Score", "RMSE (log scale)", "MAE (log scale)"))
    for col, metric in enumerate(["R2", "RMSE", "MAE"], start=1):
        vals = [METRICS[m][metric] for m in models]
        fig.add_trace(
            go.Bar(
                x=models, y=vals, marker_color=[MODEL_COLORS[m] for m in models],
                text=[f"{v:.4f}" for v in vals], textposition="outside",
                hovertemplate="%{x}<br>" + metric + ": %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col,
        )
    fig.update_annotations(font=dict(color="white", size=12))
    fig.update_xaxes(tickangle=-20)
    return style_fig(fig, height=380, showlegend=False)


def fig_actual_vs_predicted(df, model_name):
    color = MODEL_COLORS[model_name]
    customdata, template = hover_for(df)
    mn = float(min(df["Actual"].min(), df["Predicted"].min()))
    mx = float(max(df["Actual"].max(), df["Predicted"].max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Actual"], y=df["Predicted"], mode="markers", name="Houses",
        marker=dict(color=color, size=7, opacity=0.55, line=dict(width=0)),
        customdata=customdata, hovertemplate=template,
    ))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect prediction",
        line=dict(color="white", dash="dash", width=1.5), hoverinfo="skip",
    ))
    fig.update_xaxes(title="Actual Sale Price ($)")
    fig.update_yaxes(title="Predicted Sale Price ($)")
    return style_fig(fig, title=f"{model_name} — Actual vs Predicted", height=460)


def fig_all_models_grid():
    models = list(DFRAME_MAP.keys())
    fig = make_subplots(rows=2, cols=2, subplot_titles=models, horizontal_spacing=0.09, vertical_spacing=0.18)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (r, c), m in zip(positions, models):
        df = DFRAME_MAP[m]
        customdata, template = hover_for(df)
        color = MODEL_COLORS[m]
        mn = float(min(df["Actual"].min(), df["Predicted"].min()))
        mx = float(max(df["Actual"].max(), df["Predicted"].max()))
        fig.add_trace(go.Scatter(
            x=df["Actual"], y=df["Predicted"], mode="markers", showlegend=False,
            marker=dict(color=color, size=5, opacity=0.5),
            customdata=customdata, hovertemplate=template,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color="white", dash="dash", width=1),
        ), row=r, col=c)
    fig.update_annotations(font=dict(color="white", size=12))
    fig.update_xaxes(title_text="Actual ($)", row=2)
    fig.update_yaxes(title_text="Predicted ($)", col=1)
    return style_fig(fig, height=640, showlegend=False)


def fig_residual_panels(df, model_name, threshold, is_anomaly, bins):
    color = MODEL_COLORS[model_name]
    residuals = df["Residual"].values
    customdata, template = hover_for(df)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Residual Plot", "Residual Distribution"))

    fig.add_trace(go.Scatter(
        x=df["Predicted"][~is_anomaly], y=residuals[~is_anomaly], mode="markers", name="Normal",
        marker=dict(color=color, size=6, opacity=0.45),
        customdata=customdata[~is_anomaly], hovertemplate=template,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Predicted"][is_anomaly], y=residuals[is_anomaly], mode="markers", name="Anomaly",
        marker=dict(color="#ff4b4b", size=10, opacity=0.9, line=dict(color="white", width=0.5)),
        customdata=customdata[is_anomaly], hovertemplate=template,
    ), row=1, col=1)
    fig.add_hline(y=threshold, line=dict(color="#2ecc71", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=-threshold, line=dict(color="#2ecc71", dash="dash", width=1), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="white", width=0.5), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=bins, marker_color=color, opacity=0.75, name="Residuals", showlegend=False,
        hovertemplate="Residual range: %{x}<br>Count: %{y}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(x=threshold, line=dict(color="#ff4b4b", dash="dash", width=1.5), row=1, col=2)
    fig.add_vline(x=-threshold, line=dict(color="#ff4b4b", dash="dash", width=1.5), row=1, col=2)
    fig.add_vline(x=0, line=dict(color="white", width=1), row=1, col=2)

    fig.update_xaxes(title_text="Predicted Sale Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Residual ($)", row=1, col=1)
    fig.update_xaxes(title_text="Residual ($)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_annotations(font=dict(color="white", size=12))
    return style_fig(fig, height=460)


def fig_anomaly_counts():
    models = ["Linear Regression", "Random Forest", "Decision Tree", "SVR"]
    counts = [len(IDX_MAP[m]) for m in models]
    order = sorted(range(len(models)), key=lambda i: counts[i])
    models = [models[i] for i in order]
    counts = [counts[i] for i in order]
    fig = go.Figure(go.Bar(
        x=counts, y=models, orientation="h",
        marker_color=[MODEL_COLORS[m] for m in models],
        text=counts, textposition="outside",
        hovertemplate="%{y}: %{x} anomalies<extra></extra>",
    ))
    fig.update_xaxes(title="Number of Anomalies", range=[0, max(counts) + 3])
    return style_fig(fig, title="Anomalies Detected per Model", height=320, showlegend=False)


def fig_overlap_heatmap():
    models = ["Linear Regression", "Random Forest", "Decision Tree", "SVR"]
    sets = {m: set(IDX_MAP[m]) for m in models}
    z = [[len(sets[m1] & sets[m2]) for m2 in models] for m1 in models]
    fig = go.Figure(go.Heatmap(
        z=z, x=models, y=models,
        colorscale=[[0, "#1e2130"], [1, "#4e8cff"]],
        text=z, texttemplate="%{text}", textfont=dict(color="white"),
        hovertemplate="%{y} \u2229 %{x}: %{z} shared houses<extra></extra>",
        showscale=False,
    ))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, title="Anomaly Overlap Between Models", height=380, showlegend=False)


def fig_robust_split(total, robust):
    fig = go.Figure(go.Pie(
        labels=["Robust (2+ models)", "Single-model only"],
        values=[robust, total - robust],
        marker=dict(colors=["#ffa500", "#4e8cff"]),
        hole=0.55,
        hovertemplate="%{label}: %{value} houses<extra></extra>",
        textfont=dict(color="white"),
    ))
    return style_fig(fig, title="Robust vs Single-Model Anomalies", height=340)


def fig_r2_glance():
    models = sorted(METRICS.keys(), key=lambda m: METRICS[m]["R2"])
    vals = [METRICS[m]["R2"] for m in models]
    fig = go.Figure(go.Bar(
        x=vals, y=models, orientation="h",
        marker_color=[MODEL_COLORS[m] for m in models],
        text=[f"{v:.4f}" for v in vals], textposition="outside",
        customdata=[[METRICS[m]["RMSE"], METRICS[m]["MAE"]] for m in models],
        hovertemplate="%{y}<br>R\u00b2: %{x:.4f}<br>RMSE: %{customdata[0]:.4f}<br>MAE: %{customdata[1]:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="R\u00b2 Score", range=[0, 1])
    return style_fig(fig, title="R\u00b2 at a Glance", height=320, showlegend=False)


def fig_prediction_comparison(house_idx, actual):
    models = list(DFRAME_MAP.keys())
    preds, resids = [], []
    for m in models:
        row = DFRAME_MAP[m][DFRAME_MAP[m]["House_Index"] == house_idx]
        preds.append(float(row["Predicted"].values[0]) if len(row) else 0.0)
        resids.append(float(row["Residual"].values[0]) if len(row) else 0.0)

    fig = go.Figure(go.Bar(
        x=models, y=preds, marker_color=[MODEL_COLORS[m] for m in models],
        text=[f"${p:,.0f}" for p in preds], textposition="outside",
        customdata=resids,
        hovertemplate="%{x}<br>Predicted: $%{y:,.0f}<br>Residual: $%{customdata:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=actual, line=dict(color="white", dash="dash", width=1.5),
                  annotation_text=f"Actual: ${actual:,.0f}", annotation_font_color="white")
    fig.update_yaxes(title="Price ($)")
    return style_fig(fig, title=f"House #{house_idx} — Model Predictions vs Actual", height=440, showlegend=False)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", [
        "Overview",
        "Model Performance",
        "Residual Analysis",
        "Anomaly Detection",
        "House Explorer",
    ])
    st.markdown("---")
    st.markdown("**Project**")
    st.caption("A Comparative Study of Model-Dependent Residual-Based Anomaly Detection Using Multiple Predictive Models")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption(f"Ames Housing • {len(rf_df)} test houses • 4 models")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("A Comparative Study of Model-Dependent Residual-Based Anomaly Detection")
    st.caption("Ames Housing Dataset  •  4 Predictive Models  •  Residual-Based Anomaly Detection")

    st.markdown("---")
    st.markdown("### What this project does")
    st.markdown("""
    We train **4 different regression models** to predict house sale prices.  
    Each model's **prediction errors (residuals)** are then analysed — houses with unusually large residuals are flagged as **anomalies**.  
    The key question: *do different models agree on which houses are anomalous?*
    """)

    st.markdown('<div class="section-header">Quick Stats</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test Houses", f"{len(rf_df):,}")
    c2.metric("Models Compared", "4")
    c3.metric("Best R²", "0.9365", "Linear Regression")
    c4.metric("Total Unique Anomalies", len(set(SVR_IDX+RF_IDX+LR_IDX+DT_IDX)))

    # find robust anomalies (≥2 models)
    all_idx = set(SVR_IDX)|set(RF_IDX)|set(LR_IDX)|set(DT_IDX)
    robust = [i for i in all_idx if sum([i in SVR_IDX, i in RF_IDX, i in LR_IDX, i in DT_IDX]) >= 2]
    c5.metric("Robust Anomalies (≥2 models)", len(robust))

    st.markdown('<div class="section-header">Model Snapshot</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig_r2_glance(), use_container_width=True)
    with col_b:
        st.plotly_chart(fig_robust_split(len(all_idx), len(robust)), use_container_width=True)

    st.markdown('<div class="section-header">Model Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([
        {"Model": m, "R²": v["R2"], "RMSE (log)": v["RMSE"], "MAE (log)": v["MAE"],
         "Anomalies Detected": len(IDX_MAP[m])}
        for m, v in METRICS.items()
    ]).sort_values("R²", ascending=False).reset_index(drop=True)
    st.dataframe(summary.style.format({"R²":"{:.4f}","RMSE (log)":"{:.4f}","MAE (log)":"{:.4f}"}), use_container_width=True)

    st.markdown('<div class="section-header">Methodology</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1 — Train Models**")
        st.caption("Each model trained on 80% of cleaned Ames Housing data (2,927 houses). SalePrice log-transformed before training.")
    with col2:
        st.markdown("**Step 2 — Compute Residuals**")
        st.caption("Residual = Actual Price − Predicted Price. Houses where |Residual| > 3×std(Residuals) are flagged as anomalies.")
    with col3:
        st.markdown("**Step 3 — Compare**")
        st.caption("Anomalies flagged by 2+ models are called Robust Anomalies — these are the most likely genuine anomalies in the dataset.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance Comparison")

    st.plotly_chart(fig_metric_bars(), use_container_width=True)

    st.markdown("---")
    show_all = st.checkbox("Compare all models side-by-side", value=False)

    if show_all:
        st.markdown("### Actual vs Predicted — All Models")
        st.caption("Click a legend item, or hover any point, on the single-model view for full detail. Tighter clouds around the dashed line mean better fit.")
        st.plotly_chart(fig_all_models_grid(), use_container_width=True)
    else:
        st.markdown("### Actual vs Predicted")
        model_choice = st.selectbox("Select model", list(METRICS.keys()))
        df_sel = DFRAME_MAP[model_choice]

        st.plotly_chart(fig_actual_vs_predicted(df_sel, model_choice), use_container_width=True)

        m = METRICS[model_choice]
        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{m['R2']:.4f}")
        c2.metric("RMSE (log)", f"{m['RMSE']:.4f}")
        c3.metric("MAE (log)", f"{m['MAE']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Residual Analysis":
    st.title("Residual Analysis")
    st.caption("Residual = Actual Price − Predicted Price. Large residuals indicate the model struggled to predict that house.")

    top1, top2, top3 = st.columns([2, 1, 1])
    with top1:
        model_choice = st.selectbox("Select model", list(METRICS.keys()))
    with top2:
        sigma = st.slider("Threshold (× σ)", 1.0, 5.0, 3.0, 0.25)
    with top3:
        bins = st.slider("Histogram bins", 10, 80, 40, 5)

    df_sel = DFRAME_MAP[model_choice]
    residuals = df_sel["Residual"].values
    threshold = sigma * np.std(residuals)
    is_anomaly = np.abs(residuals) > threshold

    c1, c2, c3 = st.columns(3)
    c1.metric("Std of Residuals", f"${np.std(residuals):,.0f}")
    c2.metric(f"Threshold ({sigma:g}σ)", f"${threshold:,.0f}")
    c3.metric("Anomalies at this threshold", f"{is_anomaly.sum()}")
    if sigma != 3.0:
        st.caption("This slider is for exploring sensitivity only — the official anomaly list used elsewhere in the dashboard is fixed at 3σ.")

    st.plotly_chart(fig_residual_panels(df_sel, model_choice, threshold, is_anomaly, bins), use_container_width=True)

    if is_anomaly.sum() > 0:
        st.markdown("### Anomalous Houses")
        anom_df = df_sel[is_anomaly][["House_Index","Actual","Predicted","Residual"]].copy()
        anom_df = anom_df.sort_values("Residual", key=abs, ascending=False)
        anom_df["Actual"] = anom_df["Actual"].apply(lambda x: f"${x:,.0f}")
        anom_df["Predicted"] = anom_df["Predicted"].apply(lambda x: f"${x:,.0f}")
        anom_df["Residual"] = anom_df["Residual"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(anom_df.reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Detection":
    st.title("Anomaly Detection Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Anomalies Detected per Model")
        st.plotly_chart(fig_anomaly_counts(), use_container_width=True)
    with col2:
        st.markdown("### Where Models Agree")
        st.plotly_chart(fig_overlap_heatmap(), use_container_width=True)

    st.markdown("---")
    st.markdown("### All Detected Anomalies")

    all_idx = set(SVR_IDX)|set(RF_IDX)|set(LR_IDX)|set(DT_IDX)
    rows = []
    for idx in sorted(all_idx):
        models_detecting = [m for m in IDX_MAP if idx in IDX_MAP[m]]
        actual = rf_df[rf_df["House_Index"]==idx]["Actual"].values
        rows.append({
            "House Index": idx,
            "Actual Price": f"${actual[0]:,.0f}" if len(actual)>0 else "N/A",
            "Detected By (# models)": len(models_detecting),
            "Model List": models_detecting,
            "Models": ", ".join(models_detecting),
        })

    filt1, filt2 = st.columns(2)
    with filt1:
        model_filter = st.multiselect("Filter by model", list(IDX_MAP.keys()), default=list(IDX_MAP.keys()))
    with filt2:
        min_models = st.slider("Minimum models agreeing", 1, 4, 1)

    filtered = [r for r in rows if r["Detected By (# models)"] >= min_models and any(m in model_filter for m in r["Model List"])]
    if filtered:
        anomaly_table = pd.DataFrame(filtered).drop(columns=["Model List"]).sort_values("Detected By (# models)", ascending=False)
    else:
        anomaly_table = pd.DataFrame(columns=["House Index", "Actual Price", "Detected By (# models)", "Models"])
    st.dataframe(anomaly_table.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("### Robust Anomalies (Flagged by 2+ Models)")
    st.caption("These are the most reliable anomalies — multiple models agree they are unusual.")

    robust_rows = [r for r in rows if r["Detected By (# models)"] >= 2]
    robust_df = pd.DataFrame(robust_rows).sort_values("Detected By (# models)", ascending=False) if robust_rows else pd.DataFrame()

    for _, row in robust_df.iterrows():
        with st.expander(f"House #{row['House Index']} — {row['Actual Price']} — Flagged by {row['Detected By (# models)']} models"):
            st.markdown(f"**Models that flagged this house:** {row['Models']}")
            hi = row["House Index"]
            for m, df_m in DFRAME_MAP.items():
                if hi in IDX_MAP[m]:
                    row_data = df_m[df_m["House_Index"]==hi]
                    if len(row_data):
                        resid = row_data["Residual"].values[0]
                        pred  = row_data["Predicted"].values[0]
                        st.markdown(f"- **{m}**: Predicted `${pred:,.0f}` → Residual `${resid:,.0f}`")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HOUSE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "House Explorer":
    st.title("House Explorer")
    st.caption("Pick any house from the test set and see how each model predicted its price.")

    all_anom_idx = sorted(set(SVR_IDX)|set(RF_IDX)|set(LR_IDX)|set(DT_IDX))
    mode = st.radio("Browse", ["All houses", "Anomalies only"], horizontal=True)
    if mode == "Anomalies only" and all_anom_idx:
        house_idx = st.selectbox("Select anomalous house", all_anom_idx, format_func=lambda i: f"House #{i}")
    else:
        lo, hi = int(rf_df["House_Index"].min()), int(rf_df["House_Index"].max())
        house_idx = st.slider("Select House Index", lo, hi, lo)

    actual = rf_df[rf_df["House_Index"]==house_idx]["Actual"].values[0]
    st.markdown(f"### House #{house_idx} — Actual Price: **${actual:,.0f}**")

    # Check if anomaly
    is_robust = sum([house_idx in SVR_IDX, house_idx in RF_IDX,
                     house_idx in LR_IDX, house_idx in DT_IDX]) >= 2
    if is_robust:
        st.markdown('<span class="robust-badge">Robust Anomaly — flagged by 2+ models</span>', unsafe_allow_html=True)
    elif any([house_idx in SVR_IDX, house_idx in RF_IDX, house_idx in LR_IDX, house_idx in DT_IDX]):
        st.markdown('<span class="anomaly-badge">Anomaly — flagged by 1 model</span>', unsafe_allow_html=True)

    st.markdown("---")

    cols = st.columns(4)
    for col, (mname, df_m) in zip(cols, DFRAME_MAP.items()):
        row_data = df_m[df_m["House_Index"]==house_idx]
        if len(row_data):
            pred = row_data["Predicted"].values[0]
            resid = row_data["Residual"].values[0]
            flagged = house_idx in IDX_MAP[mname]
            with col:
                st.markdown(f"**{mname}**")
                st.metric("Predicted", f"${pred:,.0f}", f"${resid:,.0f} residual")
                if flagged:
                    st.markdown('<span class="anomaly-badge">Anomaly</span>', unsafe_allow_html=True)

    # Bar chart comparison
    st.markdown("### Prediction Comparison")
    st.plotly_chart(fig_prediction_comparison(house_idx, actual), use_container_width=True)