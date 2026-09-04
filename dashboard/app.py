import streamlit as st
from utils.kpis import compute_kpis
from utils.data_loader import load_data
from utils.clustering import get_cluster_pca_df
import plotly.express as px
from utils.alerts import compute_alerts
from utils.charts import (
    plot_volume_bar,
    plot_sentiment_by_issue,
    plot_complexity_agent_mismatch,
    compute_mismatch_count,
    compute_worst_sentiment_area,
    plot_word_count_box,
    compute_top_volume_area,
    compute_word_count_stats,
)

st.set_page_config(page_title="BrownBox Support Analytics", layout="wide")

st.markdown("""
<style>
div[class*="st-key-card-"] {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.8rem;
}
h2 {
    margin-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)

df = load_data()

st.title("BrownBox Customer Support Analytics")

if "filter_version" not in st.session_state:
    st.session_state.filter_version = 0

def reset_filters():
    st.session_state.filter_version += 1

st.sidebar.header("Filters")
st.sidebar.button("↺ Reset filters", on_click=reset_filters)

v = st.session_state.filter_version
issue_areas = st.sidebar.multiselect(
    "Issue Area", options=sorted(df["issue_area"].unique()), key=f"filter_issue_area_{v}"
)
sentiments = st.sidebar.multiselect(
    "Customer Sentiment", options=sorted(df["customer_sentiment"].unique()), key=f"filter_sentiment_{v}"
)
complexities = st.sidebar.multiselect(
    "Issue Complexity", options=sorted(df["issue_complexity"].unique()), key=f"filter_complexity_{v}"
)

# Empty multiselect = no filter applied on that field (show everything).
filtered = df.copy()
if issue_areas:
    filtered = filtered[filtered["issue_area"].isin(issue_areas)]
if sentiments:
    filtered = filtered[filtered["customer_sentiment"].isin(sentiments)]
if complexities:
    filtered = filtered[filtered["issue_complexity"].isin(complexities)]

preview_n = min(50, len(filtered))
with st.expander(f"🔍 Preview filtered data (showing {preview_n} of {len(filtered):,} filtered, {len(df):,} total)"):
    st.dataframe(filtered.head(50), use_container_width=True)

# --- Tabs: one per narrative section (KPIs+Alerts → Volume → Sentiment/Complexity → Distribution → Clusters) ---
tab_kpi, tab_volume, tab_sentiment, tab_wordcount, tab_cluster = st.tabs([
    "📊 KPIs & Alerts",
    "📦 Ticket Volume",
    "😐 Sentiment & Complexity",
    "📏 Word Count",
    "🧩 Ticket Clusters",
])

# --- Tab 1: KPI cards + Alerts ---
with tab_kpi:
    st.header("Key Metrics")
    with st.container(border=True, key="card-kpi"):
        kpis = compute_kpis(filtered)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tickets", f"{kpis['total_tickets']:,}")
        col2.metric("% Negative / Frustrated", f"{kpis['pct_negative']}%")
        col3.metric("% High Complexity", f"{kpis['pct_high_complexity']}%")
        col4.metric("Top Product Category", kpis['top_category'])

    st.header("Alerts")
    alerts = compute_alerts(filtered, df)
    severity_style = {
        "critical": st.error,
        "warning": st.warning,
        "ok": st.success,
    }
    with st.container(border=True, key="card-alerts"):
        alert_cols = st.columns(len(alerts))
        for col, alert in zip(alert_cols, alerts):
            with col:
                severity_style[alert["severity"]](
                    f"**{alert['label']}**\n\n{alert['value']}\n\n_{alert['baseline']}_"
                )

# --- Tab 2: Volume ---
with tab_volume:
    st.header("Ticket Volume")
    with st.container(border=True, key="card-volume"):
        st.plotly_chart(plot_volume_bar(filtered), use_container_width=True)

        top_area, top_count, top_pct = compute_top_volume_area(filtered)
        if top_area:
            st.write(
                f"📌 **{top_area}** is the largest category in the current view "
                f"at **{top_count} tickets ({top_pct:.0f}%)**."
            )

# --- Tab 3: Sentiment & Complexity ---
with tab_sentiment:
    st.header("Sentiment & Complexity Breakdown")
    with st.container(border=True, key="card-sentiment"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_sentiment_by_issue(filtered), use_container_width=True)
            worst_area, worst_rate, worst_n = compute_worst_sentiment_area(filtered)
            if worst_area:
                st.write(
                    f"⚠️ **{worst_area}** has the highest negative/frustrated rate "
                    f"at **{worst_rate * 100:.0f}%** ({worst_n} tickets)."
                )
            elif not filtered.empty:
                st.caption(
                    "No issue area has enough tickets in this view (15+) to "
                    "reliably call out a sentiment leader."
                )
        with col_b:
            st.plotly_chart(plot_complexity_agent_mismatch(filtered), use_container_width=True)
            mismatch_count = compute_mismatch_count(filtered)
            MISMATCH_MIN_N = 5  # below this, one-off tickets read as noise, not a pattern

            if mismatch_count == 0:
                st.write(
                    "✅ No high-complexity tickets in this view are handled by a "
                    "junior/inexperienced agent."
                )
            elif mismatch_count < MISMATCH_MIN_N:
                st.write(
                    f"📌 **{mismatch_count} ticket{'s' if mismatch_count != 1 else ''}** "
                    f"in this view are high-complexity and handled by a junior/"
                    f"inexperienced agent — too few on their own to call a pattern, "
                    f"but worth watching if it keeps turning up."
                )
            else:
                st.write(
                    f"⚠️ **{mismatch_count} tickets** are high-complexity and handled by a "
                    f"junior/inexperienced agent — that combination is a training/staffing "
                    f"risk, not just a data pattern."
                )

# --- Tab 4: Word Count / Distribution & Outliers ---
with tab_wordcount:
    st.header("Word Count Distribution")
    with st.container(border=True, key="card-wordcount"):
        st.plotly_chart(plot_word_count_box(filtered), use_container_width=True)

        wc_stats = compute_word_count_stats(filtered)
        st.write(
            f"📌 Median ticket length is **{wc_stats['median']:.0f} words**; "
            f"**{wc_stats['n_outliers']} tickets ({wc_stats['pct_outliers']:.1f}%)** "
            f"run unusually long, past **{wc_stats['upper_bound']:.0f} words** "
            f"(1.5×IQR)."
        )

# --- Tab 5: Clusters ---
with tab_cluster:
    st.header("Ticket Clusters")

    # NEW: short, plain-language discussion for the audience (consultant ask #4).
    # The existing methodology expander below stays as-is for anyone who wants
    # the technical silhouette/ARI detail — this is the readable version.
    st.markdown(
        "**What these groupings mean:** each point is one support ticket, "
        "placed by how similar its conversation text is to other tickets — "
        "so tickets clustering together tend to describe similar problems "
        "in similar language, without ever being told the category. Six of "
        "the eight groupings line up cleanly with a single issue type "
        "(**Warranty Claims** and **Login & Account Access** separate "
        "almost perfectly), which is a good sanity check: text alone "
        "recovers most of the real taxonomy. Two groupings, marked "
        "**(Mixed)**, don't have one dominant topic — that's where order, "
        "shipping, and installation tickets use overlapping language "
        "regardless of the actual issue. Treat those two as lower-"
        "confidence and don't over-read their exact boundary."
    )

    with st.expander("ℹ️ Methodology & Limitations"):
        st.write(
            "Silhouette scores are low across k=2–10 (0.027–0.048), consistent "
            "with TF-IDF conversation text not separating into tight, well-"
            "defined clusters at this dataset size. k=6 was tried first as a "
            "domain-informed default (issue_area already has 6 real "
            "categories), after two preprocessing fixes tested against a plain "
            "baseline (silhouette 0.043, ARI 0.349, NMI 0.479): capping terms "
            "appearing in over 60% of tickets, and removing agent first names "
            "plus boilerplate phrases (thank you, please, sure) that were "
            "leaking into top per-cluster terms. Both fixes were validated "
            "against real issue_area labels (ARI/NMI), not silhouette alone — "
            "a third option, reducing dimensionality via SVD before "
            "clustering, raised silhouette further (0.055) but was rejected "
            "after it substantially hurt agreement with issue_area and "
            "collapsed half the dataset into one incoherent cluster, concrete "
            "proof silhouette alone can mislead. k=7 and k=8 were then tested "
            "with the same winning preprocessing: k=8 was adopted for a higher "
            "ARI (0.405 vs. 0.372 at k=6) and because the extra clusters split "
            "a broad 'Returns & Refunds' bucket into two genuinely distinct, "
            "interpretable sub-themes — exchanges/pickup returns vs. refund/"
            "bank processing — rather than fragmenting arbitrarily; silhouette "
            "was actually slightly lower at k=8 (0.049) than k=6 (0.051), so "
            "it was not the deciding factor. Cluster-vs-issue_area crosstabs "
            "show six of eight clusters align with one dominant category — "
            "Warranty Claims and Login & Account Access separate especially "
            "cleanly (~98% each). Two clusters, covering Shopping/Order/"
            "Installation and Shipping Options, have no single dominant "
            "category and are labeled '(Mixed)' rather than assigned a false "
            "single-topic name — treat those two groupings as lower-confidence."
        )

    cluster_df = get_cluster_pca_df(df)

    fig = px.scatter(
        cluster_df,
        x="pc1",
        y="pc2",
        color="topic",
        hover_data={
            "conversation_preview": True,
            "issue_area": True,
            "pc1": False,
            "pc2": False,
        },
        title="Support Tickets — PCA Projection of KMeans Clusters",
        labels={"pc1": "PC1", "pc2": "PC2"},
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))

    with st.container(border=True, key="card-cluster"):
        st.plotly_chart(fig, use_container_width=True)