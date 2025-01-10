import streamlit as st
import pandas as pd
import numpy as np

# -------------
# CONFIG
# -------------
BOOKS_DIST_CSV = "books_distances.csv"
MOVIES_DIST_CSV = "movies_distances.csv"
BOOKS_CSV = "books_popular_ocean.csv"
MOVIES_CSV = "movies_popular_ocean.csv"

OCEAN_COLS = [
    "mean_Openness", 
    "mean_Conscientiousness", 
    "mean_Extraversion", 
    "mean_Agreeableness", 
    "mean_Neuroticism"
]

# -------------
# APP START
# -------------

def load_data():
    # Distances
    df_books_dist = pd.read_csv(BOOKS_DIST_CSV)
    df_movies_dist = pd.read_csv(MOVIES_DIST_CSV)

    # Original OCEAN info
    df_books = pd.read_csv(BOOKS_CSV)
    df_movies = pd.read_csv(MOVIES_CSV)
    
    return df_books, df_movies, df_books_dist, df_movies_dist

@st.cache_data
def get_item_distances(item_name, df_dist):
    """
    Given an item name and a DF of distances (long form),
    return a dataframe with columns [item2, distance],
    sorted by ascending distance.
    """
    subset = df_dist[df_dist["item1"] == item_name].copy()
    subset.sort_values(by="distance", ascending=True, inplace=True)
    return subset[["item2", "distance"]]


def main():
    st.title("Personality-Based Book/Movie Similarity")

    # Load all data
    df_books, df_movies, df_books_dist, df_movies_dist = load_data()

    # ---------------------------------------
    # Side Panel: Selections
    # ---------------------------------------
    st.sidebar.header("Search Settings")

    # 1) Choose base item type: Book or Movie
    item_type = st.sidebar.radio(
        "Pick type of base item:",
        ("Books", "Movies")
    )

    # 2) Choose which item from that type
    if item_type == "Books":
        all_items = df_books["item"].unique()
    else:  # "Movies"
        all_items = df_movies["item"].unique()
    
    selected_item = st.sidebar.selectbox(
        f"Select a {item_type[:-1]} from the list:",  # "Book" or "Movie"
        sorted(all_items)
    )

    # 3) Number of items to display (N)
    N = st.sidebar.slider("Number of items to show:", min_value=1, max_value=20, value=5)

    # 4) Similar or Different?
    similarity_mode = st.sidebar.radio(
        "Do you want the most similar or the most different items?",
        ("Similar", "Different")
    )

    # 5) Which types of items to compare against?
    compare_options = st.sidebar.multiselect(
        "Compare against which categories?",
        ["Books", "Movies"],
        default=["Books", "Movies"]
    )

    # ---------------------------------------
    # MAIN LOGIC
    # ---------------------------------------
    # Retrieve relevant distance DF 
    if item_type == "Books":
        df_dist = df_books_dist
    else:
        df_dist = df_movies_dist

    # Get all distances from the selected item
    dist_subset = get_item_distances(selected_item, df_dist)

    # If user wants to see the most different items, sort descending
    if similarity_mode == "Different":
        dist_subset = dist_subset.sort_values(by="distance", ascending=False)

    # Now filter to items from the desired categories
    # If the user wants to see Books only, we filter 'item2' that is in df_books['item'] 
    # If the user wants to see Movies only, we filter 'item2' that is in df_movies['item'] 
    # If both, we keep all
    valid_items_books = set(df_books["item"])
    valid_items_movies = set(df_movies["item"])

    def is_valid(item2):
        is_book = item2 in valid_items_books
        is_movie = item2 in valid_items_movies
        if "Books" in compare_options and is_book:
            return True
        if "Movies" in compare_options and is_movie:
            return True
        return False

    dist_subset = dist_subset[dist_subset["item2"].apply(is_valid)]

    # Finally, take the top N results
    dist_subset = dist_subset.head(N)

    # ---------------------------------------
    # DISPLAY
    # ---------------------------------------
    st.write(f"### Results for {item_type[:-1]}: **{selected_item}**")
    if similarity_mode == "Similar":
        st.write(f"Showing **{N}** most similar items among **{compare_options}**.")
    else:
        st.write(f"Showing **{N}** most different items among **{compare_options}**.")

    # Display results table
    st.dataframe(dist_subset.rename(columns={"item2": "Item", "distance": "Distance"}))

    st.write("You can adjust the selection in the sidebar to see different results.")


if __name__ == "__main__":
    main()
