#data_loader.py
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "conversations_preprocessed.parquet"

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Loads the preprocessed support conversations dataset (997 rows, includes
    conversation_clean and word_count from the notebook's Day 5/Day 4 work).

    @st.cache_data matters here: Streamlit reruns the ENTIRE script top-to-bottom
    on every widget interaction (every filter click, every dropdown change). Without
    caching, that means re-reading the parquet file from disk on every single click.
    Caching keys off the function's inputs (none here) and file contents, so it only
    re-reads if the file itself changes.
    """
    return pd.read_parquet(DATA_PATH)