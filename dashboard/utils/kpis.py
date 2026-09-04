#kpis.py
import pandas as pd

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Computes the 4 headline KPIs from a (possibly filtered) tickets dataframe.
    Plain function, not cached — aggregating ~1000 rows is instant, and
    caching adds overhead without benefit here.
    """
    total_tickets = len(df)

    pct_negative = (
        df["customer_sentiment"].isin(["negative", "frustrated"]).mean() * 100
        if total_tickets else 0.0
    )

    pct_high_complexity = (
        (df["issue_complexity"] == "high").mean() * 100
        if total_tickets else 0.0
    )

    top_category = (
        df["product_category"].mode().iloc[0] if total_tickets else "N/A"
    )

    return {
        "total_tickets": total_tickets,
        "pct_negative": round(pct_negative, 1),
        "pct_high_complexity": round(pct_high_complexity, 1),
        "top_category": top_category,
    }