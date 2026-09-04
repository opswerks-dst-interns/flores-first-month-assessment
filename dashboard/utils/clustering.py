#clustering.py
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

AGENT_NAMES = {
    "sarah", "john", "alex", "rachel", "jane", "emily", "jack", "tom",
    "lisa", "mark", "lily", "david", "sam", "mike", "kate", "anna", "jake",
}

CONTRACTION_FRAGMENTS = {
    "im", "youre", "youll", "youve", "dont", "didnt", "isnt",
    "wasnt", "wouldnt", "couldnt", "cant", "ive", "theyre", "thats",
}

BOILERPLATE_STOPWORDS = {
    "thank", "please", "sure", "let", "know", "okay", "welcome",
    "great", "day", "anything", "else", "moment",
}

EXTRA_STOPWORDS = AGENT_NAMES | BOILERPLATE_STOPWORDS | CONTRACTION_FRAGMENTS


def vectorize_conversations(df, max_features=3000, min_df=2, max_df=0.6, stop_words=None):
    """
    Fits a NEW TfidfVectorizer on the dashboard's conversation_clean column.

    max_df=0.6 + stop_words=EXTRA_STOPWORDS (passed by the caller) were
    tested against a plain baseline via silhouette / ARI / NMI (ARI/NMI
    compare cluster assignments to real issue_area labels, so they're the
    more meaningful check than silhouette alone):
      - Baseline:                    silhouette=0.043, ARI=0.349, NMI=0.479
      - max_df=0.6 only:              silhouette=0.045, ARI=0.357, NMI=0.504
      - max_df=0.6 + EXTRA_STOPWORDS: silhouette=0.051, ARI=0.372, NMI=0.509 (at k=6)

    A TruncatedSVD dimensionality-reduction step was also tested and
    rejected: it pushed silhouette higher (0.055) but hurt ARI (0.198) and
    NMI (0.344), and collapsed ~51% of the dataset into one incoherent
    cluster — concrete proof silhouette alone can mislead, which is why it
    isn't trusted in isolation anywhere in this pipeline.

    Resulting vocabulary with max_df + stopwords: 1,854 terms — under the
    max_features=3000 cap, so max_features is no longer the binding
    constraint. Left in place as a harmless ceiling.

    See explore_clustering.py for the full comparison run and crosstabs.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=list(stop_words) if stop_words else None,
    )
    X = vectorizer.fit_transform(df["conversation_clean"])
    return X, vectorizer


def run_kmeans(X, n_clusters, random_state=42):
    """
    Fits KMeans on the TF-IDF matrix and returns cluster labels + the fitted
    model. random_state is fixed so cluster ids stay stable run-to-run,
    which is what makes CLUSTER_LABELS below a valid, reusable mapping.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km

K_SELECTED = 8

CLUSTER_LABELS = {
    0: "Order Status & Cancellations",
    1: "Shopping, Installation & Order (Mixed)",
    2: "Shipping Options (Mixed)",
    3: "Delivery Delays",
    4: "Exchanges & Pickup Returns",
    5: "Login & Account Access",
    6: "Warranty Claims",
    7: "Returns & Refund Processing",
}


@st.cache_data
def get_cluster_pca_df(df, n_clusters=K_SELECTED, random_state=42):
    """
    Runs the full clustering pipeline and returns a plain DataFrame
    ready for plotting: one row per ticket, with its cluster id,
    topic label, and 2D PCA coordinates.
    """
    X, _ = vectorize_conversations(df, stop_words=EXTRA_STOPWORDS)
    labels, _ = run_kmeans(X, n_clusters=n_clusters, random_state=random_state)

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X.toarray())

    plot_df = df.copy()
    plot_df["cluster"] = labels
    plot_df["topic"] = plot_df["cluster"].map(CLUSTER_LABELS)
    plot_df["pc1"] = coords[:, 0]
    plot_df["pc2"] = coords[:, 1]
    plot_df["conversation_preview"] = (
        plot_df["conversation"].str.slice(0, 150) + "…"
    )
    return plot_df