import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
from rapidfuzz import fuzz

BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

OCEAN_COLS = [
    "mean_Openness", 
    "mean_Conscientiousness", 
    "mean_Extraversion", 
    "mean_Agreeableness", 
    "mean_Neuroticism"
]

from rapidfuzz import fuzz

def fuzzy_deduplicate_items(df, col="item", distance_threshold=2):
    """
    Collapse near-duplicate strings in `df[col]` into canonical forms.
    We'll treat two items as duplicates if their
    Levenshtein distance <= distance_threshold.
    
    Steps:
      1. Collect unique items and sort them.
      2. Maintain a list of canonical items.
      3. For each new item, if it's within distance_threshold of an
         existing canonical item, map it to that canonical form.
         Otherwise, add it as a new canonical item.
      4. Remap all items in df to their canonical forms.
      5. Optionally drop duplicates if you want only one row per canonical item.
    """
    items = list(df[col].unique())
    items.sort()

    canonical_items = []            # List of distinct 'master' strings
    item_to_canonical = {}          # Maps each original string -> canonical string

    for it in items:
        matched_canonical = None
        for c in canonical_items:
            # rapidfuzz.distance.Levenshtein.distance(it, c)
            dist = fuzz.distance(it, c)  # defaults to Levenshtein distance
            if dist <= distance_threshold:
                matched_canonical = c
                break
        
        if matched_canonical is not None:
            # We found a canonical item close enough to 'it'
            item_to_canonical[it] = matched_canonical
        else:
            # Make 'it' a new canonical item
            canonical_items.append(it)
            item_to_canonical[it] = it

    # Now transform df[col] to use the canonical forms
    df[col] = df[col].apply(lambda x: item_to_canonical[x])
    
    # If you want to keep only one row per canonical item, you can drop duplicates now
    # or later. This keeps the first occurrence of each canonical item.
    df.drop_duplicates(subset=col, keep="first", inplace=True)

    return df
    
@st.cache_data
def load_data():
    """Load & fuzzy-deduplicate books/movies."""
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)

    # Convert to string, strip whitespace
    df_books["item"] = df_books["item"].fillna("").astype(str).str.strip()
    df_movies["item"] = df_movies["item"].fillna("").astype(str).str.strip()

    # Drop empty
    df_books = df_books[df_books["item"] != ""]
    df_movies = df_movies[df_movies["item"] != ""]

    # Fuzzy deduplicate using the function above
    df_books = fuzzy_deduplicate_items(df_books, col="item", distance_threshold=2)
    df_movies = fuzzy_deduplicate_items(df_movies, col="item", distance_threshold=2)

    return df_books, df_movies

def get_item_vector(df, item_name):
    row = df.loc[df["item"] == item_name]
    if row.empty:
        return None
    return row[OCEAN_COLS].values[0]  # shape: (5,)

def compute_distances(base_vector, df):
    vectors = df[OCEAN_COLS].values
    base_vector_2d = base_vector.reshape(1, -1)
    distances = euclidean_distances(base_vector_2d, vectors)[0]
    return pd.DataFrame({"item": df["item"], "distance": distances})

def main():
    st.title("Personality-Based Book/Movie Similarity (Fuzzy Deduped)")

    df_books, df_movies = load_data()

    st.sidebar.header("Search Settings")

    item_type = st.sidebar.radio("Base item type:", ("Books", "Movies"))

    if item_type == "Books":
        valid_items = sorted(df_books["item"].unique())
        df_base = df_books
    else:
        valid_items = sorted(df_movies["item"].unique())
        df_base = df_movies

    selected_item = st.sidebar.selectbox(f"Select a {item_type[:-1]}:", valid_items)
    N = st.sidebar.slider("Number of items to show:", 1, 20, 5)
    similarity_mode = st.sidebar.radio("Similar or Different?", ("Similar", "Different"))
    compare_options = st.sidebar.multiselect("Compare with:", ["Books", "Movies"], default=["Books","Movies"])

    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"No OCEAN vector for '{selected_item}'")
        return

    result_frames = []
    if "Books" in compare_options:
        dist_books = compute_distances(base_vector, df_books)
        dist_books = dist_books[dist_books["item"] != selected_item]
        dist_books["category"] = "Book"
        result_frames.append(dist_books)

    if "Movies" in compare_options:
        dist_movies = compute_distances(base_vector, df_movies)
        dist_movies = dist_movies[dist_movies["item"] != selected_item]
        dist_movies["category"] = "Movie"
        result_frames.append(dist_movies)

    if not result_frames:
        st.warning("No categories selected.")
        return

    df_results = pd.concat(result_frames, ignore_index=True)
    ascending = (similarity_mode == "Similar")
    df_results.sort_values("distance", ascending=ascending, inplace=True)
    df_top = df_results.head(N)

    st.subheader(f"Selected {item_type[:-1]}: {selected_item}")
    st.dataframe(df_top[["item", "category", "distance"]])

    # Build a multi-bar chart for each of the top items
    df_plot_rows = []
    for _, row in df_top.iterrows():
        i_name, cat = row["item"], row["category"]
        if cat == "Book":
            df_source = df_books
        else:
            df_source = df_movies

        item_row = df_source[df_source["item"] == i_name]
        if item_row.empty:
            continue

        ocean_vals = item_row[OCEAN_COLS].iloc[0]
        for trait_col in OCEAN_COLS:
            trait_label = trait_col.replace("mean_", "")
            df_plot_rows.append({
                "Item": i_name,
                "Trait": trait_label,
                "Score": ocean_vals[trait_col]
            })

    df_plot = pd.DataFrame(df_plot_rows)
    if not df_plot.empty:
        fig = px.bar(
            df_plot,
            x="Trait",
            y="Score",
            color="Item",
            barmode="group",
            title="OCEAN Trait Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("Use the sidebar to change settings.")

if __name__ == "__main__":
    main()
