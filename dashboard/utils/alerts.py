#alerts.py
def compute_alerts(filtered_df, baseline_df):
    """
    Returns a list of alert dicts, each comparing a rate in the filtered
    slice against the full-dataset baseline rate. Severity is dynamic:
    it's driven by how far the filtered rate deviates from baseline, not
    a hardcoded number — so alerts stay meaningful under any filter combo.
    """
    alerts = []
    total = len(filtered_df)

    # --- Alert 1: negative/frustrated sentiment rate vs baseline ---
    baseline_negative_rate = baseline_df["customer_sentiment"].isin(
        ["negative", "frustrated"]
    ).mean()
    filtered_negative_rate = (
        filtered_df["customer_sentiment"].isin(["negative", "frustrated"]).mean()
        if total else 0.0
    )
    delta = filtered_negative_rate - baseline_negative_rate
    alerts.append({
        "label": "Negative/Frustrated Sentiment",
        "value": f"{filtered_negative_rate * 100:.1f}%",
        "baseline": f"{baseline_negative_rate * 100:.1f}% baseline",
        "severity": "critical" if delta > 0.10 else "warning" if delta > 0.03 else "ok",
    })

    # --- Alert 2: high-complexity + junior/inexperienced agent mismatch ---
    mismatch_mask = (filtered_df["issue_complexity"] == "high") & (
        filtered_df["agent_experience_level"].isin(["junior", "inexperienced"])
    )
    mismatch_count = int(mismatch_mask.sum())
    mismatch_rate = mismatch_count / total if total else 0.0
    baseline_mismatch_rate = (
        (baseline_df["issue_complexity"] == "high")
        & (baseline_df["agent_experience_level"].isin(["junior", "inexperienced"]))
    ).mean()
    delta = mismatch_rate - baseline_mismatch_rate
    alerts.append({
        "label": "High-Complexity / Inexperienced-Agent Mismatch",
        "value": f"{mismatch_count} tickets ({mismatch_rate * 100:.1f}%)",
        "baseline": f"{baseline_mismatch_rate * 100:.1f}% baseline",
        "severity": "critical" if delta > 0.05 else "warning" if delta > 0.02 else "ok",
    })

    # --- Alert 3: word count outlier rate (long/drawn-out tickets) vs baseline ---
    q1 = baseline_df["word_count"].quantile(0.25)
    q3 = baseline_df["word_count"].quantile(0.75)
    upper_bound = q3 + 1.5 * (q3 - q1)  # bound fixed from baseline, not the filtered slice
    baseline_outlier_rate = (baseline_df["word_count"] > upper_bound).mean()
    filtered_outlier_rate = (
        (filtered_df["word_count"] > upper_bound).mean() if total else 0.0
    )
    delta = filtered_outlier_rate - baseline_outlier_rate
    alerts.append({
        "label": "Long/Drawn-Out Tickets (word count outliers)",
        "value": f"{filtered_outlier_rate * 100:.1f}%",
        "baseline": f"{baseline_outlier_rate * 100:.1f}% baseline",
        "severity": "critical" if delta > 0.05 else "warning" if delta > 0.02 else "ok",
    })

    return alerts