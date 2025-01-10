import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# -----------------------------------
# CONFIG / CONSTANTS
# -----------------------------------
BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

# Column names for the personality embeddings
OCEAN_COLS = [
    "mean_Openness", 
    "mean_Conscientiousness", 
    "mean_Extraversion", 
    "mean_Agreeableness", 
    "mean_Neuroticism"
]

# -----------------------------------
# DATA LOADING & CACHE
# -----------------------------------
@st.cache_data
def load_data():
    """Load books and movies DataFrames."""
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)
    return df_books, df_movies

def get_item_vector(df, item_name):
    """
    Given a DataFrame (books or movies) and an item_name,
    return the OCEAN embedding as a 1D numpy array.
    If item not found, returns None.
    """
    row = df.loc[df["item"] == item_name]
    if row.empty:
        return None
    return row[OCEAN_COLS].values[0]  # [0] to get as 1D

def compute_distances(base_vector, df):
    """
    For each item in df, compute Euclidean distance 
    from base_vector, return a new DataFrame with columns:
    [item, distance].
    """
    # Extract all vectors in df
    vectors = df[OCEAN_COLS].values  # shape (num_items, 5)
    
    # Reshape base_vector to (1, 5) so sklearn can handle it
    base_vector_2d = base_vector.reshape(1, -1)

    # Compute distances for each item
    distances = euclidean_distances(base_vector_2d, vectors)[0]  # shape (num_items,)

    # Build a DataFrame
    dist_df = pd.DataFrame({
        "item": df["item"],
        "distance": distances
    })
    return dist_df

# -----------------------------------
# STREAMLIT APP
# -----------------------------------
def main():
    st.title("Personality-Based Book/Movie Similarity (Lightweight)")

    # Load the CSV data
    df_books, df_movies = load_data()

    st.sidebar.header("Search Settings")

    # 1) Choose the base item type: Book or Movie
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Build the list of items from that type
    if item_type == "Books":
        all_items = sorted(df_books["item"].unique())
        df_base = df_books
    else:
        all_items = sorted(df_movies["item"].unique())
        df_base = df_movies

    # 3) Choose which item from that type
    selected_item = st.sidebar.selectbox(
        f"Select a {item_type[:-1]}:",
        all_items
    )

    # 4) Number of items to display
    N = st.sidebar.slider("Number of items to show:", min_value=1, max_value=20, value=5)

    # 5) Similar or Different
    similarity_mode = st.sidebar.radio(
        "Show most similar or most different?",
        ("Similar", "Different")
    )

    # 6) Which types of items to compare against?
    compare_options = st.sidebar.multiselect(
        "Compare against which categories?",
        ["Books", "Movies"],
        default=["Books", "Movies"]
    )

    # --------------------------------------
    # COMPUTE ON THE FLY
    # --------------------------------------
    # 1) Get the selected item's vector
    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"Could not find {selected_item} in {item_type}.")
        return

    # 2) For each category in compare_options, compute distances
    result_frames = []
    if "Books" in compare_options:
        # compute distance from selected_item -> all books
        dist_books = compute_distances(base_vector, df_books)
        # drop the item itself if it's in that set
        dist_books = dist_books[dist_books["item"] != selected_item]
        # tag them as "Book"
        dist_books["category"] = "Book"
        result_frames.append(dist_books)
    if "Movies" in compare_options:
        # compute distance from selected_item -> all movies
        dist_movies = compute_distances(base_vector, df_movies)
        dist_movies = dist_movies[dist_movies["item"] != selected_item]
        dist_movies["category"] = "Movie"
        result_frames.append(dist_movies)

    # Combine them
    if len(result_frames) == 0:
        st.warning("No categories selected to compare against.")
        return
    df_results = pd.concat(result_frames, ignore_index=True)

    # 3) Sort by ascending distance for "similar," or descending for "different"
    ascending = True if similarity_mode == "Similar" else False
    df_results.sort_values(by="distance", ascending=ascending, inplace=True)

    # 4) Take top N
    df_top = df_results.head(N)

    # --------------------------------------
    # DISPLAY
    # --------------------------------------
    st.subheader(f"Selected {item_type[:-1]}: {selected_item}")
    if similarity_mode == "Similar":
        st.write(f"Showing **{N}** most similar items from: {', '.join(compare_options)}")
    else:
        st.write(f"Showing **{N}** most different items from: {', '.join(compare_options)}")

    # Display in table
    # You can rename columns or add more info if you like
    st.dataframe(df_top[["item", "category", "distance"]])

    st.write("Use the sidebar to change settings.")

if __name__ == "__main__":
    main()
