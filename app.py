import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px

# Import the Levenshtein distance function for fuzzy deduplication
from rapidfuzz.distance import Levenshtein

# ------------------------------------------------------------------
# CONFIG / CONSTANTS
# ------------------------------------------------------------------
BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

# OCEAN columns (already normalized in your data)
OCEAN_COLS = [
    "mean_Openness", 
    "mean_Conscientiousness", 
    "mean_Extraversion", 
    "mean_Agreeableness", 
    "mean_Neuroticism"
]

# ------------------------------------------------------------------
# FUZZY DEDUPLICATION UTILITY
# ------------------------------------------------------------------
def fuzzy_deduplicate_items(df, col="item", distance_threshold=2):
    """
    Collapse near-duplicate strings in `df[col]` into canonical forms.
    We'll treat two items as duplicates if their Levenshtein distance <= distance_threshold.
    
    Steps:
      1. Collect unique items and sort them.
      2. Maintain a list of canonical items.
      3. For each new item, if it's within distance_threshold of an
         existing canonical item, map it to that canonical form.
         Otherwise, add it as a new canonical item.
      4. Remap all items in df to their canonical forms.
      5. Drop duplicates so each final item name appears only once.
    """
    items = list(df[col].unique())
    items.sort()

    canonical_items = []
    item_to_canonical = {}

    for it in items:
        matched_canonical = None
        for c in canonical_items:
            dist = Levenshtein.distance(it, c)
            if dist <= distance_threshold:
                matched_canonical = c
                break
        
        if matched_canonical is not None:
            item_to_canonical[it] = matched_canonical
        else:
            canonical_items.append(it)
            item_to_canonical[it] = it

    df[col] = df[col].apply(lambda x: item_to_canonical[x])
    df.drop_duplicates(subset=col, keep="first", inplace=True)

    return df

# ------------------------------------------------------------------
# DATA LOADING & PREPARATION
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load books and movies, ensuring 'item' is deduplicated (fuzzy).
    """
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)

    # Convert 'item' to string, fill NaNs, strip whitespace
    df_books["item"] = df_books["item"].fillna("").astype(str).str.strip()
    df_movies["item"] = df_movies["item"].fillna("").astype(str).str.strip()

    # Drop empty items
    df_books = df_books[df_books["item"] != ""]
    df_movies = df_movies[df_movies["item"] != ""]

    # Fuzzy deduplicate
    df_books = fuzzy_deduplicate_items(df_books, col="item", distance_threshold=2)
    df_movies = fuzzy_deduplicate_items(df_movies, col="item", distance_threshold=2)

    return df_books, df_movies

def get_item_vector(df, item_name):
    row = df.loc[df["item"] == item_name]
    if row.empty:
        return None
    return row[OCEAN_COLS].values[0]

def compute_distances(base_vector, df):
    vectors = df[OCEAN_COLS].values
    base_vector_2d = base_vector.reshape(1, -1)
    distances = euclidean_distances(base_vector_2d, vectors)[0]
    return pd.DataFrame({"item": df["item"], "distance": distances})

# ------------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------------
def main():
    st.title("Book/Movie Similarity by Big Five Scores")

    # --- Load data (cached) ---
    df_books, df_movies = load_data()

    mode = st.sidebar.radio(
        "Choose a Mode:",
        ["Find Similar Items", "Sort by Trait"]
    )

    if mode == "Find Similar Items":
        show_find_similar_items(df_books, df_movies)
    else:
        show_sort_by_trait(df_books, df_movies)

def show_find_similar_items(df_books, df_movies):
    """Existing functionality: pick an item, find most similar/different."""
    st.sidebar.header("Find Similar Items")

    # 1) Choose base item type: Book or Movie
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Build the dropdown of valid items (sorted)
    if item_type == "Books":
        valid_items = sorted(df_books["item"].unique())
        df_base = df_books
    else:
        valid_items = sorted(df_movies["item"].unique())
        df_base = df_movies

    selected_item = st.sidebar.selectbox(f"Select a {item_type[:-1]}:", valid_items)

    # 3) Number of items to show
    N = st.sidebar.slider("Number of items to show:", min_value=1, max_value=20, value=5)

    # 4) Similar or Different
    similarity_mode = st.sidebar.radio(
        "Show most similar or most different?",
        ("Similar", "Different")
    )

    # 5) Compare to Books, Movies, or Both
    compare_options = st.sidebar.multiselect(
        "Compare against which categories?",
        ["Books", "Movies"],
        default=["Books", "Movies"]
    )

    # --- Main logic ---

    # Get the base item vector
    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"Could not find '{selected_item}' in '{item_type}'.")
        return

    result_frames = []

    if "Books" in compare_options:
        dist_books = compute_distances(base_vector, df_books)
        # Use plural "Books" for category
        dist_books = dist_books[dist_books["item"] != selected_item]
        dist_books["category"] = "Books"
        result_frames.append(dist_books)

    if "Movies" in compare_options:
        dist_movies = compute_distances(base_vector, df_movies)
        # Use plural "Movies" for category
        dist_movies = dist_movies[dist_movies["item"] != selected_item]
        dist_movies["category"] = "Movies"
        result_frames.append(dist_movies)

    if not result_frames:
        st.warning("No categories selected to compare against.")
        return

    df_results = pd.concat(result_frames, ignore_index=True)

    # Sort ascending for "Similar" or descending for "Different"
    ascending = (similarity_mode == "Similar")
    df_results.sort_values("distance", ascending=ascending, inplace=True)

    # Take top N
    df_top = df_results.head(N)

    st.subheader(f"Selected {item_type[:-1]}: {selected_item}")
    if similarity_mode == "Similar":
        st.write(f"Showing **{N}** most similar items from: {', '.join(compare_options)}.")
    else:
        st.write(f"Showing **{N}** most different items from: {', '.join(compare_options)}.")

    if df_top.empty:
        st.warning("No results to display in df_top!")
        return

    st.dataframe(df_top[["item", "category", "distance"]])

    # Build a multi-bar chart for the top items
    df_plot_rows = []
    for _, row in df_top.iterrows():
        item_name = row["item"]
        category = row["category"]

        if category == "Books":
            df_source = df_books
        else:
            df_source = df_movies

        item_row = df_source[df_source["item"] == item_name]
        if item_row.empty:
            continue

        ocean_scores = item_row[OCEAN_COLS].iloc[0]
        for trait_col in OCEAN_COLS:
            trait_label = trait_col.replace("mean_", "")
            df_plot_rows.append({
                "Item": item_name,
                "Trait": trait_label,
                "Score": ocean_scores[trait_col]
            })

    df_plot = pd.DataFrame(df_plot_rows)
    if not df_plot.empty:
        fig = px.bar(
            df_plot,
            x="Trait",
            y="Score",
            color="Item",
            barmode="group",
            hover_name="Item",
            title="OCEAN Trait Comparison (Top Matches)"
        )
        fig.update_layout(xaxis_title="OCEAN Trait", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)


def show_sort_by_trait(df_books, df_movies):
    """New functionality: sort items by a chosen trait."""
    st.sidebar.header("Sort by Personality Trait")

    # 1) Choose which trait to sort by
    trait = st.sidebar.selectbox(
        "Select a trait to rank by:",
        OCEAN_COLS, 
        format_func=lambda x: x.replace("mean_", "")  # nicer display
    )
    
    # 2) Highest or Lowest
    order_mode = st.sidebar.radio(
        "Order by trait value:",
        ["Highest", "Lowest"]
    )
    ascending = (order_mode == "Lowest")

    # 3) Number of items to show
    N = st.sidebar.slider("Number of items to display:", min_value=1, max_value=50, value=10)

    # 4) Data source(s): Books, Movies, or Both
    data_options = st.sidebar.multiselect(
        "Which data sources?",
        ["Books", "Movies"],
        default=["Books", "Movies"]
    )

    st.subheader("Sort by Trait Results")

    # Combine or filter books/movies as requested
    df_books_mod = df_books.copy()
    df_books_mod["category"] = "Books"
    df_movies_mod = df_movies.copy()
    df_movies_mod["category"] = "Movies"

    df_combined = pd.concat([df_books_mod, df_movies_mod], ignore_index=True)

    # Filter by chosen data sources
    df_filtered = df_combined[df_combined["category"].isin(data_options)]

    if df_filtered.empty:
        st.warning("No data remains after filtering for categories.")
        return

    # Sort by the chosen trait
    df_sorted = df_filtered.sort_values(by=trait, ascending=ascending)
    df_top = df_sorted.head(N)

    if df_top.empty:
        st.warning("No items to show in the top slice!")
        return

    st.write(f"Showing the {order_mode.lower()} **{N}** items based on **{trait.replace('mean_','')}**.")
    st.dataframe(df_top[["item", "category", trait]])

    # Build a multi-bar chart for these top items
    df_plot_rows = []
    for _, row in df_top.iterrows():
        item_name = row["item"]
        category = row["category"]
        
        # We already have the OCEAN columns in row
        ocean_scores = row[OCEAN_COLS]

        for trait_col in OCEAN_COLS:
            trait_label = trait_col.replace("mean_", "")
            df_plot_rows.append({
                "Item": item_name,
                "Trait": trait_label,
                "Score": ocean_scores[trait_col],
                "Category": category
            })

    df_plot = pd.DataFrame(df_plot_rows)
    if not df_plot.empty:
        fig = px.bar(
            df_plot,
            x="Trait",
            y="Score",
            color="Item",
            barmode="group",
            hover_name="Item",
            title="OCEAN Trait Comparison (Ranked Results)"
        )
        fig.update_layout(xaxis_title="OCEAN Trait", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
