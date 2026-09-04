#charts.py
import plotly.express as px

ISSUE_COMPLEXITY_ORDER = ["less", "medium", "high"]
AGENT_EXPERIENCE_ORDER = ["inexperienced", "junior", "experienced"]
SENTIMENT_COLORS = {
    "positive": "#2ECC71",    # green
    "neutral": "#95A5A6",     # gray
    "negative": "#FF7700",    # red
    "frustrated": "#FF1900",  # darker red — worse than plain negative
}

AGENT_EXPERIENCE_COLORS = {
    "inexperienced": "#AED6F1",  # light blue
    "junior": "#3498DB",         # blue
    "experienced": "#1B4F72",    # dark blue
}

def plot_volume_bar(df):
    """Ticket count per issue_area — answers 'where's the volume concentrated'."""
    counts = df["issue_area"].value_counts().reset_index()
    counts.columns = ["issue_area", "count"]
    fig = px.bar(
        counts, x="issue_area", y="count",
        title="Ticket Volume by Issue Area",
        labels={"issue_area": "Issue Area", "count": "Tickets"},
        text_auto=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_sentiment_by_issue(df):
    grouped = (
        df.groupby(["issue_area", "customer_sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        grouped, x="issue_area", y="count", color="customer_sentiment",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS,
        title="Customer Sentiment by Issue Area",
        labels={"issue_area": "Issue Area", "count": "Tickets"},
        text_auto=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_complexity_agent_mismatch(df):
    """
    Grouped bar: issue_complexity x agent_experience_level.
    The 'mismatch' signal is specifically high-complexity tickets
    landing with inexperienced/junior agents — flagged separately
    below the chart, not just left implicit in the bars.
    """
    grouped = (
        df.groupby(["issue_complexity", "agent_experience_level"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        grouped, x="issue_complexity", y="count", color="agent_experience_level",
        barmode="group",
        category_orders={
            "issue_complexity": ISSUE_COMPLEXITY_ORDER,
            "agent_experience_level": AGENT_EXPERIENCE_ORDER,
        },
        color_discrete_map=AGENT_EXPERIENCE_COLORS,
        title="Issue Complexity vs. Agent Experience Level",
        labels={"issue_complexity": "Issue Complexity", "count": "Tickets"},
        text_auto=True,
    )
    fig.update_traces(textposition="outside")
    return fig


def compute_mismatch_count(df):
    """High-complexity tickets handled by a non-experienced agent — the risk pattern."""
    mask = (df["issue_complexity"] == "high") & (
        df["agent_experience_level"].isin(["junior", "inexperienced"])
    )
    return int(mask.sum())

def compute_worst_sentiment_area(df, min_tickets=15):
    """
    Issue area with the highest negative/frustrated rate — restricted to
    areas with at least `min_tickets` tickets, so a single unlucky ticket
    (100% of n=1) can't outrank a real, sizeable rate elsewhere.

    Ties in rate are broken by ticket volume: two areas at the same rate
    aren't equally "worst" if one represents 4 affected tickets and the
    other represents 20 — the larger one reflects more actual impact.

    Returns (area, rate, n) or (None, 0.0, 0) if no area clears the floor.
    """
    if df.empty:
        return None, 0.0, 0
    negative_mask = df["customer_sentiment"].isin(["negative", "frustrated"])
    grouped = (
        df.assign(is_negative=negative_mask)
        .groupby("issue_area")
        .agg(rate=("is_negative", "mean"), n=("is_negative", "size"))
    )
    eligible = grouped[grouped["n"] >= min_tickets]
    if eligible.empty:
        return None, 0.0, 0
    ranked = eligible.sort_values(by=["rate", "n"], ascending=[False, False])
    worst_area = ranked.index[0]
    return worst_area, float(ranked.iloc[0]["rate"]), int(ranked.iloc[0]["n"])

def plot_word_count_box(df):
    """
    Box plot of word_count with individual outlier points shown.
    IQR bounds are computed here (not hardcoded) so the annotation
    stays correct even as filters change the underlying slice.
    """
    fig = px.box(
        df, y="word_count", points="outliers",
        title="Ticket Word Count Distribution",
        labels={"word_count": "Word Count"},
    )
    q1 = df["word_count"].quantile(0.25)
    q3 = df["word_count"].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    n_outliers = int((df["word_count"] > upper_bound).sum())
    fig.add_annotation(
        text=f"{n_outliers} tickets above {upper_bound:.0f} words (1.5×IQR)",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, align="right",
        font=dict(size=11),
    )
    return fig

def compute_top_volume_area(df):
    """
    Issue area with the highest ticket count in the current slice, plus its
    share of total volume — the volume-chart equivalent of the worst-
    sentiment-area / mismatch-count callouts already used on the sentiment
    and complexity charts. Returns (area, count, pct) or (None, 0, 0.0) on
    an empty filtered slice.
    """
    if df.empty:
        return None, 0, 0.0
    counts = df["issue_area"].value_counts()
    top_area = counts.idxmax()
    top_count = int(counts.max())
    pct = top_count / len(df) * 100
    return top_area, top_count, pct


def compute_word_count_stats(df):
    """
    Median word count + IQR-outlier rate for the current slice. Reuses the
    same 1.5xIQR bound as plot_word_count_box's on-chart annotation, so the
    written discussion below the chart can never disagree with what the
    chart itself is showing. Returns zeros on an empty filtered slice.
    """
    if df.empty:
        return {"median": 0, "upper_bound": 0, "n_outliers": 0, "pct_outliers": 0.0}
    median = df["word_count"].median()
    q1 = df["word_count"].quantile(0.25)
    q3 = df["word_count"].quantile(0.75)
    upper_bound = q3 + 1.5 * (q3 - q1)
    n_outliers = int((df["word_count"] > upper_bound).sum())
    pct_outliers = n_outliers / len(df) * 100
    return {
        "median": median,
        "upper_bound": upper_bound,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers,
    }