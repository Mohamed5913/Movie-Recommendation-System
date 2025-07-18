import streamlit as st
import pandas as pd
import numpy as np
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Load Data
u_data = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
u_item = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=[0, 1, 2], names=["movie_id", "title", "release_date"])
u_user = pd.read_csv("ml-100k/u.user", sep="|", header=None, names=["user_id", "age", "gender", "occupation", "zip_code"])
u_genre = pd.read_csv("ml-100k/u.genre", sep="|", names=["genre", "genre_id"], engine='python')

# Preprocessing
movie_genres = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1", header=None, usecols=range(5, 24))
genre_cols = u_genre.genre.dropna().tolist()
u_item = u_item.join(movie_genres)
u_item.columns = list(u_item.columns[:3]) + genre_cols

# Initialize user_movie_matrix with 5 users (user 1 with actual data, others empty)
full_matrix = u_data.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
user_movie_matrix = pd.DataFrame(0, index=range(1, 6), columns=full_matrix.columns)
user_movie_matrix.loc[1] = full_matrix.loc[1]

def get_cosine_sim():
    return cosine_similarity(user_movie_matrix)

@st.cache_data
def get_movie_posters(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
    response = requests.get(url)
    data = response.json()
    return data.get("Poster") if data.get("Response") == "True" else None

# UI
st.title("🎥 Movie Recommendation System")

# --- Select User ---
user_list = user_movie_matrix.index.tolist()
selected_user = st.selectbox("👤 Select User ID", user_list)

# --- Add New User ---
st.subheader("➕ Add New User")
new_user_name = st.text_input("Enter New User Name")
if st.button("Add User"):
    if new_user_name:
        new_user_id = user_movie_matrix.index.max() + 1
        user_movie_matrix.loc[new_user_id] = 0
        st.success(f"User '{new_user_name}' added successfully!")
    else:
        st.warning("Please enter a user name")

# --- Rate a Movie ---
st.subheader("🎬 Rate a Movie")

# Filter by genre
selected_genres = st.multiselect("Filter by Genre", genre_cols)

# Filter by release year
u_item['release_year'] = pd.to_datetime(u_item['release_date'], errors='coerce').dt.year
min_year = int(u_item['release_year'].min())
max_year = int(u_item['release_year'].max())
year_range = st.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

# Search for movie
search_query = st.text_input("Search for a Movie")
filtered_movies = u_item.copy()
if selected_genres:
    for genre in selected_genres:
        filtered_movies = filtered_movies[filtered_movies[genre] == 1]
filtered_movies = filtered_movies[filtered_movies['release_year'].between(year_range[0], year_range[1])]
if search_query:
    filtered_movies = filtered_movies[filtered_movies["title"].str.contains(search_query, case=False, na=False)]

movie_titles = filtered_movies["title"].tolist()
selected_movie = st.selectbox("Select a Movie to Rate", movie_titles)
rating = st.slider("Your Rating", 1, 5, 3)
if st.button("Give Rating"):
    movie_id = u_item[u_item["title"] == selected_movie]["movie_id"].values[0]
    user_movie_matrix.loc[selected_user, movie_id] = rating
    st.success(f"You rated '{selected_movie}' a {rating}/5")

# --- Get Recommendations ---
st.subheader("🎯 Get Movie Recommendations")
num_recs = st.slider("Number of Recommendations", 5, 20, 10)
if st.button("Recommend"):
    cosine_sim = get_cosine_sim()
    sim_scores = cosine_sim[selected_user - 1]

    sim_users = [(i + 1, score) for i, score in enumerate(sim_scores) if (i + 1) != selected_user]
    sim_users = sorted(sim_users, key=lambda x: x[1], reverse=True)

    sim_user_ids, sim_weights = zip(*sim_users)

    ratings_matrix = user_movie_matrix.loc[sim_user_ids]
    weighted_ratings = ratings_matrix.T.dot(np.array(sim_weights)) / (np.sum(sim_weights) + 1e-8)

    seen_movies = user_movie_matrix.loc[selected_user] > 0
    recommendations = pd.Series(weighted_ratings, index=ratings_matrix.columns)
    recommendations = recommendations[~seen_movies].sort_values(ascending=False).head(num_recs)

    st.write("### Recommended Movies:")
    for movie_id in recommendations.index:
        title = u_item[u_item["movie_id"] == movie_id]["title"].values[0]
        poster_url = get_movie_posters(title)
        if poster_url:
            st.image(poster_url, width=100)
        st.write(f"**{title}** - Predicted Rating: {recommendations[movie_id]:.2f}")

# --- Sidebar Data Exploration ---
st.sidebar.title("📊 Data Exploration")
if st.sidebar.checkbox("Show raw data"):
    st.write("### Raw Ratings Data", u_data.head())
    st.write("### Raw Movie Data", u_item.head())
    st.write("### Raw User Data", u_user.head())

if st.sidebar.checkbox("Show ratings distribution"):
    fig, ax = plt.subplots()
    sns.countplot(x="rating", data=u_data, ax=ax)
    st.pyplot(fig)

if st.sidebar.checkbox("Top Rated Movies"):
    top_movies = u_data.groupby("item_id")["rating"].mean().sort_values(ascending=False).head(10)
    top_titles = u_item[u_item["movie_id"].isin(top_movies.index)]["title"]
    st.write(pd.DataFrame({"Title": top_titles.values, "Avg Rating": top_movies.values}))