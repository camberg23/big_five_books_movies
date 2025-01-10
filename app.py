import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# ------------------------------------------------------------------
# CONFIG / CONSTANTS
# ------------------------------------------------------------------
BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

# Column names for personality embeddings
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

    # Convert NaNs or floats in 'item' to empty string, then ensure type str
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
    from base_vector, return a DataFrame with:
    [item, distance].
    """
    # All vectors
    vectors = df[OCEAN_COLS].values  # shape: (num_items, 5)

    # Reshape base_vector to (1, 5) for sklearn
    base_vector_2d = base_vector.reshape(1, -1)

    # Euclidean distances to each row
    distances = euclidean_distances(base_vector_2d, vectors)[0]  # shape: (num_items,)

    dist_df = pd.DataFrame({
        "item": df["item"],
        "distance": distances
    })
    return dist_df

def normalize_ocean_scores(row, global_min, global_max):
    """
    Given a Series of OCEAN values, min, and max for each trait,
    return normalized (0 to 1) scores.
    """
    return (row - global_min) / (global_max - global_min)

# ------------------------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------------------------
def main():
    st.title("Personality-Based Book/Movie Similarity")

    # --- Load data once (cached) ---
    df_books, df_movies = load_data()

    # Compute global min/max for OCEAN across both sets (for normalization)
    combined_ocean = pd.concat([df_books[OCEAN_COLS], df_movies[OCEAN_COLS]], ignore_index=True)
    global_min = combined_ocean.min()
    global_max = combined_ocean.max()

    # -----------------------
    # SIDEBAR SELECTION
    # -----------------------
    st.sidebar.header("Search Settings")

    # 1) Choose base item type: Book or Movie
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Build the list of valid items (exclude blanks)
    if item_type == "Books":
        valid_items = [x for x in df_books["item"].unique() if x.strip() != ""]
        df_base = df_books
    else:  # Movies
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
    # A) Get base item vector
    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"Could not find '{selected_item}' in '{item_type}'.")
        return

    # B) For each category in compare_options, compute distances
    result_frames = []

    if "Books" in compare_options:
        dist_books = compute_distances(base_vector, df_books)
        # Remove the item itself if it's in the same set
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

    # Show a table of the top results
    st.dataframe(df_top[["item", "category", "distance"]])

    # Plot normalized OCEAN scores for each returned item
    st.write("### OCEAN Profiles for Each Returned Item")
    for i, row in df_top.iterrows():
        item_name = row["item"]
        category = row["category"]
        dist_val = row["distance"]

        # Find that item's row in either df_books or df_movies
        if category == "Book":
            df_source = df_books
        else:
            df_source = df_movies

        row_data = df_source.loc[df_source["item"] == item_name]
        if row_data.empty:
            st.write(f"**{item_name}** (No OCEAN data found)")
            continue

        # Extract OCEAN scores
        ocean_scores = row_data[OCEAN_COLS].iloc[0]
        # Normalize
        normalized_scores = normalize_ocean_scores(ocean_scores, global_min, global_max)

        st.subheader(f"{item_name} ({category}, Distance={dist_val:.2f})")
        
        # Prepare a 1-row DataFrame for plotting with st.bar_chart
        # columns = traits, single row of normalized values
        df_chart = pd.DataFrame([normalized_scores.values], columns=normalized_scores.index)
        st.bar_chart(df_chart)

    st.write("Use the sidebar to change the settings.")

if __name__ == "__main__":
    main()
