import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Path to CSVs
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

@st.cache_data
def load_data():
    """Load books and movies data, ensuring 'item' is a string."""
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)

    # Convert NaNs or floats in 'item' column to empty strings, then ensure string type
    df_books["item"] = df_books["item"].fillna("").astype(str)
    df_movies["item"] = df_movies["item"].fillna("").astype(str)
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

    # Euclidean distances to each row in df
    distances = euclidean_distances(base_vector_2d, vectors)[0]  # shape: (num_items,)

    dist_df = pd.DataFrame({
        "item": df["item"],
        "distance": distances
    })
    return dist_df

def main():
    st.title("Personality-Based Book/Movie Similarity (Lightweight)")

    # --- Load data ---
    df_books, df_movies = load_data()

    # --- Sidebar inputs ---
    st.sidebar.header("Search Settings")

    # 1) Select base item type
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Choose which item
    if item_type == "Books":
        # Filter out empty or whitespace-only strings before sorting
        valid_books = [x for x in df_books["item"].unique() if x.strip() != ""]
        all_items = sorted(valid_books)
        df_base = df_books
    else:  # Movies
        valid_movies = [x for x in df_movies["item"].unique() if x.strip() != ""]
        all_items = sorted(valid_movies)
        df_base = df_movies

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

    # --- Main logic ---
    # Get OCEAN vector of the selected item
    base_vector = get_item_vector(df_base, selected_item)
    if base_vector is None:
        st.error(f"Could not find '{selected_item}' in '{item_type}'.")
        return

    # Compute distances on the fly for each selected category
    result_frames = []

    if "Books" in compare_options:
        dist_books = compute_distances(base_vector, df_books)
        # Remove the item itself
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

    # Combine the results
    df_results = pd.concat(result_frames, ignore_index=True)

    # Sort ascending for most similar, descending for most different
    ascending = (similarity_mode == "Similar")
    df_results.sort_values("distance", ascending=ascending, inplace=True)

    # Take top N
    df_top = df_results.head(N)

    # --- Display ---
    st.subheader(f"Selected {item_type[:-1]}: {selected_item}")

    if similarity_mode == "Similar":
        st.write(f"Showing **{N}** most similar items from: {', '.join(compare_options)}")
    else:
        st.write(f"Showing **{N}** most different items from: {', '.join(compare_options)}")

    st.dataframe(df_top[["item", "category", "distance"]])
    st.write("Use the sidebar to change settings.")

if __name__ == "__main__":
    main()
