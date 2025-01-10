import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px

# ------------------------------------------------------------------
# CONFIG / CONSTANTS
# ------------------------------------------------------------------
BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

# OCEAN columns (already normalized in your data, presumably)
OCEAN_COLS = [
    "mean_Openness", 
    "mean_Conscientiousness", 
    "mean_Extraversion", 
    "mean_Agreeableness", 
    "mean_Neuroticism"
]

# ------------------------------------------------------------------
# DATA LOADING & UTILITIES
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load books and movies data, ensuring 'item' is always a string."""
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)

    # Convert NaNs/floats in 'item' to empty strings, then ensure type str
    df_books["item"] = df_books["item"].fillna("").astype(str)
    df_movies["item"] = df_movies["item"].fillna("").astype(str)

    return df_books, df_movies

def get_item_vector(df, item_name):
    """
    Given a DataFrame (books or movies) and an item_name,
    return that row's OCEAN embedding as a 1D numpy array.
    If not found, return None.
    """
    row = df.loc[df["item"] == item_name]
    if row.empty:
        return None
    return row[OCEAN_COLS].values[0]  # shape: (5,)

def compute_distances(base_vector, df):
    """
    For each item in df, compute Euclidean distance 
    from base_vector, return a DataFrame with columns:
    [item, distance].
    """
    vectors = df[OCEAN_COLS].values  # shape: (num_items, 5)
    base_vector_2d = base_vector.reshape(1, -1)
    distances = euclidean_distances(base_vector_2d, vectors)[0]  # shape: (num_items,)
    
    dist_df = pd.DataFrame({
        "item": df["item"],
        "distance": distances
    })
    return dist_df

# ------------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------------
def main():
    st.title("Personality-Based Book/Movie Similarity")

    # --- Load data (cached) ---
    df_books, df_movies = load_data()

    # -----------------------
    # SIDEBAR SELECTION
    # -----------------------
    st.sidebar.header("Search Settings")

    # 1) Choose base item type: Book or Movie
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Get valid items (exclude blanks)
    if item_type == "Books":
        valid_items = [x for x in df_books["item"].unique() if x.strip() != ""]
        df_base = df_books
    else:
        valid_items = [x for x in df_movies["item"].unique() if x.strip() != ""]
        df_base = df_movies

    # Sort them for the dropdown
    all_items = sorted(valid_items)
    selected_item = st.sidebar.selectbox(f"Select a {item_type[:-1]}:", all_items)

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

    # -----------------------
    # MAIN LOGIC
    # -----------------------
    #  A) Get the base item vector
    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"Could not find '{selected_item}' in '{item_type}'.")
        return

    #  B) Compute distances on the fly for each chosen category
    result_frames = []

    if "Books" in compare_options:
        dist_books = compute_distances(base_vector, df_books)
        # Remove the item itself if it appears in the same category
        dist_books = dist_books[dist_books["item"] != selected_item]
        dist_books["category"] = "Book"
        result_frames.append(dist_books)

    if "Movies" in compare_options:
        dist_movies = compute_distances(base_vector, df_movies)
        dist_movies = dist_movies[dist_movies["item"] != selected_item]
        dist_movies["category"] = "Movie"
        result_frames.append(dist_movies)

    if not result_frames:
        st.warning("No categories selected to compare against.")
        return

    # Combine into a single DataFrame
    df_results = pd.concat(result_frames, ignore_index=True)

    # Sort ascending for "similar" or descending for "different"
    ascending = (similarity_mode == "Similar")
    df_results.sort_values("distance", ascending=ascending, inplace=True)

    # Take top N
    df_top = df_results.head(N)

    # -----------------------
    # DISPLAY RESULTS
    # -----------------------
    st.subheader(f"Selected {item_type[:-1]}: {selected_item}")

    if similarity_mode == "Similar":
        st.write(f"Showing **{N}** most similar items from: {', '.join(compare_options)}.")
    else:
        st.write(f"Showing **{N}** most different items from: {', '.join(compare_options)}.")

    # A) Show the table
    st.dataframe(df_top[["item", "category", "distance"]])

    # B) Create a single multi-bar chart with Plotly
    #    X-axis = OCEAN traits, Y-axis = score. 
    #    Each item has one bar per trait, grouped by trait.
    df_plot_rows = []
    for _, row in df_top.iterrows():
        item_name = row["item"]
        category = row["category"]
        # Find the item in the appropriate df
        if category == "Book":
            df_source = df_books
        else:
            df_source = df_movies

        this_item_row = df_source[df_source["item"] == item_name]
        if this_item_row.empty:
            continue

        # Extract the 5 OCEAN scores
        ocean_values = this_item_row[OCEAN_COLS].iloc[0]
        # Build the row structure for each trait
        for trait_col in OCEAN_COLS:
            # Remove "mean_" prefix just for a cleaner trait label
            trait_label = trait_col.replace("mean_", "")
            df_plot_rows.append({
                "Item": item_name,
                "Trait": trait_label,
                "Score": ocean_values[trait_col]
            })

    df_plot = pd.DataFrame(df_plot_rows)

    # Generate grouped bar plot with Plotly
    if not df_plot.empty:
        fig = px.bar(
            df_plot,
            x="Trait",
            y="Score",
            color="Item",
            barmode="group",
            hover_name="Item",
            title="Comparison of OCEAN Traits across Returned Items",
        )
        fig.update_layout(xaxis_title="OCEAN Trait", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    st.write("Use the sidebar to change the settings.")

if __name__ == "__main__":
    main()
