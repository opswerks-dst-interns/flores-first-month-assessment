# BrownBox Customer Support Analytics

Month 1 Data Science internship assessment: an NLP classification notebook and an interactive Streamlit BI dashboard, both built on the [NebulaByte E-Commerce Customer Support Conversations](https://huggingface.co/datasets/NebulaByte/E-Commerce_Customer_Support_Conversations) dataset (997 usable tickets after cleaning, fictional company "BrownBox").

## What's in this repo

| Deliverable                   | Location                                    | Points |
| ----------------------------- | ------------------------------------------- | ------ |
| NLP classification notebook   | `notebook/support_nlp_classification.ipynb` | 10     |
| Streamlit analytics dashboard | `dashboard/app.py`                          | 20     |

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the notebook

```
jupyter notebook notebook/support_nlp_classification.ipynb
```

Loads the dataset via Hugging Face `datasets`, cleans it, vectorizes with TF-IDF, and trains/compares three classifiers (Naive Bayes, Logistic Regression, Linear SVM).

## Running the dashboard

From the **repo root**:

```
streamlit run dashboard/app.py
```

The dashboard reads a pre-processed parquet file (`data/conversations_preprocessed.parquet`) produced by the notebook's cleaning steps. Run the notebook at least once first if that file isn't already present. The data path in `dashboard/utils/data_loader.py` is anchored to the file's own location, so this works whether you launch Streamlit from the repo root or from inside `dashboard/`.

## Repo structure

```
data-science-internship-month1/
├── README.md
├── requirements.txt
├── data/
│   └── conversations_preprocessed.parquet
├── notebook/
│   └── support_nlp_classification.ipynb
├── dashboard/
│   ├── app.py
│   └── utils/
│       ├── data_loader.py       # parquet loading, cached
│       ├── clustering.py        # TF-IDF vectorization, KMeans, PCA
│       ├── explore_clustering.py# elbow/silhouette exploration, top terms per cluster
│       ├── kpis.py              # KPI card calculations
│       ├── charts.py            # volume / sentiment / complexity / distribution charts
│       └── alerts.py            # dynamic alert thresholds vs. baseline
```

## Key findings

**Notebook — classification**

- Linear SVM was the best-performing model of the three tested: **94.5% test accuracy, 0.947 macro-F1**.
- Macro-F1 was prioritized over plain accuracy or weighted-F1 due to ~4x class imbalance across issue categories.
- Predictive feature analysis (LinearSVC coefficients) surfaces the vocabulary each class relies on most, discussed in the notebook's feature analysis section.

**Dashboard — support ticket patterns**

- Ticket volume, sentiment, and complexity are explorable live via sidebar filters (issue area, sentiment, complexity), with a one-click reset.
- Sentiment and complexity are tracked as separate KPIs deliberately — one measures how the customer feels, the other measures how hard the ticket is to resolve; they don't move together.
- A recurring risk pattern: high-complexity tickets landing with junior/inexperienced agents. This is surfaced both as a chart annotation and as a dynamic alert.
- Unsupervised KMeans (k=6) over TF-IDF, visualized via PCA, cleanly separates topics like Login & Account Access, Warranty Claims, and Returns & Refunds. Order/Shipping/Billing topics overlap more (see Known Limitations).

## Known limitations

- **No timestamp column in the source dataset** — time-trend / response-time charts were scoped out for this reason (confirmed against the dataset's actual columns), not from lack of effort.
- **Silhouette scores are low** — 0.027–0.045 across k=2–10 tested, 0.043 at the k=6 used here. Root cause, confirmed by inspecting top TF-IDF terms per cluster: agent first names (`sarah`, `emily`, `john`) leak in as high-weight features alongside generic boilerplate (`agent`, `customer`, `thank`), diluting topic-specific signal. A custom stopword list would likely improve this, but wasn't applied — see rationale below.
- **Cluster purity varies significantly**, per the cluster-vs-`issue_area` crosstab:
  - Cluster 1 (Warranty Claims): 100/103 tickets = **97%** one category — very clean
  - Cluster 4 (Login & Account Access): 140/148 = **95%** — very clean
  - Cluster 5 (Returns & Refunds): 175/214 = **82%** — clean
  - Cluster 0 (Order Status & Installation) and Cluster 3 (Order & Delivery, General): both mixed across Order/Shopping/Cancellations/Shipping categories with no strongly dominant one — Cluster 3 is the dashboard's _largest_ cluster (352 tickets) but also one of its least topically distinct
  - Cluster 2 (Billing/Shipping, 42 tickets): the clearest low-confidence case — no category exceeds 14/42, essentially no dominant topic
- **Why this wasn't fixed:** stripping agent names would require a custom stopword/NER step, a full re-run of KMeans + PCA, and re-validation of cluster labels — real pipeline changes this close to the submission deadline, on a rubric row (Clustering & Modes) already fully scored. Documenting the finding was prioritized over re-opening a working, already-validated pipeline. This would be the first thing to address if the clustering moved toward production use.
- **Two different outlier definitions coexist by design**, not by oversight: the word-count box plot recomputes its IQR bounds live off whichever slice is currently filtered (so the plot's outlier count is always relative to what's on screen), while the Alerts section fixes its IQR bounds from the full, unfiltered dataset (so alert severity means "unusual vs. the whole dataset," not "unusual vs. whatever you just filtered to"). Both are intentional; each answers a different question.

## Rubric coverage (dashboard)

| Row                        | Status         |
| -------------------------- | -------------- |
| Clustering & Modes         | ✅             |
| KPI Prioritization         | ✅             |
| Chart Selection & Grouping | ✅             |
| Distribution & Outliers    | ✅             |
| Anomaly/Alert Highlighting | ✅             |
| UI Design & Interactivity  | ✅             |
| Repo & README              | ✅             |
| Oral Walkthrough           | delivered live |

## Screenshots

_Add screenshots of the running dashboard here before final submission — e.g. the KPI row, a chart, and the cluster scatter._

```
![KPI row](docs/screenshot-kpis.png)
![Cluster scatter](docs/screenshot-clusters.png)
```
