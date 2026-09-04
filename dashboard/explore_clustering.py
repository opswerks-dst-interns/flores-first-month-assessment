# #explore_clustering.py
# import numpy as np
# import pandas as pd
# import plotly.express as px
# from sklearn.metrics import silhouette_score
# from utils.data_loader import load_data
# from utils.clustering import vectorize_conversations, run_kmeans

# df = load_data()
# X, vectorizer = vectorize_conversations(df)

# # --- Elbow + silhouette across a range of k ---
# # Starting at 2 (silhouette is undefined at k=1 — there's nothing to separate)
# # and going to 10 as a reasonable ceiling given we only have 997 rows.
# k_range = range(2, 11)
# inertias = []
# silhouettes = []

# for k in k_range:
#     labels, km = run_kmeans(X, n_clusters=k)
#     inertias.append(km.inertia_)
#     silhouettes.append(silhouette_score(X, labels))
#     print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

# # Elbow plot
# fig1 = px.line(x=list(k_range), y=inertias, markers=True,
#                 labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
# fig1.show()

# # Silhouette plot
# fig2 = px.line(x=list(k_range), y=silhouettes, markers=True,
#                 labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Score by k")
# fig2.show()

# # --- Top terms per cluster, using k=6 (chosen above) ---
# FINAL_K = 7
# labels, km = run_kmeans(X, n_clusters=FINAL_K)
# df["cluster"] = labels

# feature_names = vectorizer.get_feature_names_out()
# top_n = 12

# print("\n--- Top terms per cluster ---")
# for cluster_id in range(FINAL_K):
#     centroid = km.cluster_centers_[cluster_id]
#     top_idx = np.argsort(centroid)[::-1][:top_n]
#     top_words = [feature_names[i] for i in top_idx]
#     size = (df["cluster"] == cluster_id).sum()
#     print(f"\nCluster {cluster_id} (n={size}): {', '.join(top_words)}")

# # --- Cross-check against issue_area, same as Day 12 ---
# print("\n--- Cluster vs issue_area crosstab ---")
# print(pd.crosstab(df["cluster"], df["issue_area"]))

# import re
# from collections import Counter

# def extract_agent_names(df, text_col="conversation_clean"):
#     """
#     Every ticket opens with the agent's self-introduction ('my name is X'),
#     which after standard cleaning collapses to a consistent 'name <agentname>'
#     pattern near the start of the text. Searching only the first ~150 chars
#     avoids false hits from unrelated later mentions like 'account name' or
#     'product name'.
#     """
#     pattern = re.compile(r"\bname (\w+)\b")
#     names = Counter()
#     for text in df[text_col].str[:150]:
#         match = pattern.search(text)
#         if match:
#             names[match.group(1)] += 1
#     return names

# # Run and inspect before trusting it:
# from utils.data_loader import load_data
# df = load_data()
# names = extract_agent_names(df)
# print(names.most_common(30))


#explore_clustering.py
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from utils.data_loader import load_data
from utils.clustering import vectorize_conversations, run_kmeans
import random

np.random.seed(42)
random.seed(42)

df = load_data()
X, vectorizer = vectorize_conversations(df)

# --- Elbow + silhouette across a range of k ---
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    labels, km = run_kmeans(X, n_clusters=k)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

fig1 = px.line(x=list(k_range), y=inertias, markers=True,
                labels={"x": "k", "y": "Inertia"}, title="Elbow Method")
fig1.show()

fig2 = px.line(x=list(k_range), y=silhouettes, markers=True,
                labels={"x": "k", "y": "Silhouette Score"}, title="Silhouette Score by k")
fig2.show()

# --- Top terms per cluster, using FINAL_K ---
FINAL_K = 8
labels, km = run_kmeans(X, n_clusters=FINAL_K)
df["cluster"] = labels

feature_names = vectorizer.get_feature_names_out()
top_n = 12

print("\n--- Top terms per cluster ---")
for cluster_id in range(FINAL_K):
    centroid = km.cluster_centers_[cluster_id]
    top_idx = np.argsort(centroid)[::-1][:top_n]
    top_words = [feature_names[i] for i in top_idx]
    size = (df["cluster"] == cluster_id).sum()
    print(f"\nCluster {cluster_id} (n={size}): {', '.join(top_words)}")

print("\n--- Cluster vs issue_area crosstab ---")
print(pd.crosstab(df["cluster"], df["issue_area"]))


# ======================================================================
# --- Comparison harness: baseline vs. max_df vs. SVD vs. stopwords ---
# ======================================================================

def evaluate(X_input, label):
    km_eval = KMeans(n_clusters=FINAL_K, random_state=42, n_init="auto")
    labels_eval = km_eval.fit_predict(X_input)
    sil = silhouette_score(X_input, labels_eval)
    ari = adjusted_rand_score(df["issue_area"], labels_eval)
    nmi = normalized_mutual_info_score(df["issue_area"], labels_eval)
    print(f"{label}: silhouette={sil:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")
    return labels_eval

def top_terms_per_cluster(X_tfidf, labels_input, names, k, top_n=12):
    for cluster_id in range(k):
        mask = labels_input == cluster_id
        centroid = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(centroid)[::-1][:top_n]
        top_words = [names[i] for i in top_idx]
        print(f"  Cluster {cluster_id} (n={mask.sum()}): {', '.join(top_words)}")

print("\n\n=== COMPARISON RUNS ===\n")

# --- Baseline: current production (no max_df, no stopwords) ---
vec_base = TfidfVectorizer(max_features=3000, min_df=2)
X_base = vec_base.fit_transform(df["conversation_clean"])
labels_base = evaluate(X_base, "Baseline (current)")

# --- Variant A: max_df only ---
vec_maxdf = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.6)
X_maxdf = vec_maxdf.fit_transform(df["conversation_clean"])
labels_maxdf = evaluate(X_maxdf, "max_df=0.6 only")

# --- Variant B: max_df + SVD(100) ---
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_maxdf)
print(f"SVD(100) explained variance: {svd.explained_variance_ratio_.sum():.1%}")
labels_svd = evaluate(X_svd, "max_df=0.6 + SVD(100)")

print("\n--- Top terms, Variant B (max_df + SVD) clusters ---")
top_terms_per_cluster(X_maxdf, labels_svd, vec_maxdf.get_feature_names_out(), FINAL_K)

# --- Verified agent-name / boilerplate / contraction stopwords ---
AGENT_NAMES = {
    "sarah", "john", "alex", "rachel", "jane", "emily", "jack", "tom",
    "lisa", "mark", "lily", "david", "sam", "mike", "kate", "anna", "jake",
}
BOILERPLATE_STOPWORDS = {
    "thank", "please", "sure", "let", "know", "okay", "welcome",
    "great", "day", "anything", "else", "moment",
}
CONTRACTION_FRAGMENTS = {
    "im", "youre", "youll", "youve", "dont", "didnt", "isnt",
    "wasnt", "wouldnt", "couldnt", "cant", "ive", "theyre", "thats",
}
EXTRA_STOPWORDS = AGENT_NAMES | BOILERPLATE_STOPWORDS | CONTRACTION_FRAGMENTS

# --- Variant C: max_df + extra stopwords ---
vec_stopwords = TfidfVectorizer(
    max_features=3000, min_df=2, max_df=0.6,
    stop_words=list(EXTRA_STOPWORDS),
)
X_stopwords = vec_stopwords.fit_transform(df["conversation_clean"])
print(f"Vocabulary size: {len(vec_stopwords.get_feature_names_out())}")
labels_stopwords = evaluate(X_stopwords, "max_df=0.6 + extra stopwords")

# --- Testing k=7 and k=8 with the SAME winning vectorizer config ---
# (max_df=0.6 + EXTRA_STOPWORDS — already established as the best result;
# this only tests whether a different k improves separation further)

print("\n\n=== K COMPARISON (max_df + stopwords config) ===\n")

for test_k in [6, 7, 8]:
    km_test = KMeans(n_clusters=test_k, random_state=42, n_init="auto")
    labels_test = km_test.fit_predict(X_stopwords)
    sil = silhouette_score(X_stopwords, labels_test)
    ari = adjusted_rand_score(df["issue_area"], labels_test)
    nmi = normalized_mutual_info_score(df["issue_area"], labels_test)
    print(f"k={test_k}: silhouette={sil:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")
    sizes = pd.Series(labels_test).value_counts().sort_index()
    print(f"  cluster sizes: {sizes.tolist()}")

print("\n--- Variant C cluster vs issue_area crosstab ---")
print(pd.crosstab(labels_stopwords, df["issue_area"]))

print("\n--- Top terms, Variant C (max_df + stopwords) clusters ---")
top_terms_per_cluster(X_stopwords, labels_stopwords, vec_stopwords.get_feature_names_out(), FINAL_K)

import sklearn
print(sklearn.__version__)