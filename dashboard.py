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
import re

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY") or st.secrets.get("OMDB_API_KEY")
if not OMDB_API_KEY:
    st.error("OMDb API key not set. Please ensure it is configured in Streamlit Secrets or .env file.")

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

# --- Maintain Session State for Users ---
if "user_matrix" not in st.session_state:
    st.session_state.user_matrix = user_movie_matrix.copy()
if "user_names" not in st.session_state:
    st.session_state.user_names = {i: f"User {i}" for i in st.session_state.user_matrix.index}

@st.cache_data
def get_movie_data(title):
    if not OMDB_API_KEY:
        return {}
    clean_title = re.sub(r"\s*\(\d{4}\)$", "", title.strip())
    if ", The" in clean_title:
        clean_title = "The " + clean_title.replace(", The", "")
    elif ", A" in clean_title:
        clean_title = "A " + clean_title.replace(", A", "")
    elif ", An" in clean_title:
        clean_title = "An " + clean_title.replace(", An", "")
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={clean_title}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("Response") == "True":
            return data
        else:
            return {}
    except Exception:
        return {}

st.title("\U0001F3A5 Movie Recommendation System")

# --- Add New User ---
st.subheader("\u2795 Add New User")
new_user_name = st.text_input("Enter New User Name")
if st.button("Add User"):
    if new_user_name.strip():
        new_user_id = st.session_state.user_matrix.index.max() + 1
        st.session_state.user_matrix.loc[new_user_id] = 0
        st.session_state.user_names[new_user_id] = new_user_name.strip()
        st.success(f"User '{new_user_name}' added successfully!")
    else:
        st.warning("Please enter a valid user name.")

# --- Select User ---
user_list = list(st.session_state.user_matrix.index)
user_labels = [f"{st.session_state.user_names[uid]} (ID: {uid})" for uid in user_list]
user_label_to_id = dict(zip(user_labels, user_list))
selected_label = st.selectbox("\U0001F464 Select User", user_labels)
selected_user = user_label_to_id[selected_label]

# --- Rate a Movie ---
st.subheader("\U0001F3AC Rate a Movie")
selected_genres = st.multiselect("Filter by Genre", genre_cols)
u_item['release_year'] = pd.to_datetime(u_item['release_date'], errors='coerce').dt.year
min_year = int(u_item['release_year'].min())
max_year = int(u_item['release_year'].max())
year_range = st.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

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
if selected_movie:
    movie_data = get_movie_data(selected_movie)
    poster_url = movie_data.get("Poster")
    if poster_url and poster_url != "N/A":
        st.image(poster_url, width=100)

rating = st.slider("Your Rating", 1, 5, 3)
if st.button("Give Rating"):
    movie_id = u_item[u_item["title"] == selected_movie]["movie_id"].values[0]
    st.session_state.user_matrix.loc[selected_user, movie_id] = rating
    st.success(f"You rated '{selected_movie}' a {rating}/5")

# --- Get Recommendations ---
st.subheader("\U0001F3AF Get Movie Recommendations")
num_recs = st.slider("Number of Recommendations", 5, 20, 10)
if st.button("Recommend"):
    seen_movies = st.session_state.user_matrix.loc[selected_user] > 0

    if seen_movies.any():
        user_matrix_nonzero = st.session_state.user_matrix[(st.session_state.user_matrix > 0).any(axis=1)]
        if len(user_matrix_nonzero) < 2:
            st.warning("Not enough user ratings for recommendations.")
        else:
            cosine_sim = cosine_similarity(user_matrix_nonzero)
            user_index = list(user_matrix_nonzero.index).index(selected_user)
            sim_scores = cosine_sim[user_index]

            sim_users = [(uid, sim_scores[i]) for i, uid in enumerate(user_matrix_nonzero.index) if uid != selected_user]
            sim_users = sorted(sim_users, key=lambda x: x[1], reverse=True)

            if sim_users:
                sim_user_ids, sim_weights = zip(*sim_users)
                ratings_matrix = user_matrix_nonzero.loc[list(sim_user_ids)]
                sim_weights = np.array(sim_weights)

                weighted_ratings = []
                for movie_id in ratings_matrix.columns:
                    ratings = ratings_matrix[movie_id]
                    mask = ratings > 0
                    if mask.sum() == 0:
                        weighted_ratings.append(0.0)
                    else:
                        r = ratings[mask]
                        w = sim_weights[mask]
                        score = np.dot(r, w) / (np.sum(w) + 1e-8)
                        weighted_ratings.append(score)

                recommendations = pd.Series(weighted_ratings, index=ratings_matrix.columns)
                recommendations = recommendations[~seen_movies].sort_values(ascending=False).head(num_recs)

                st.write("### Recommended Movies:")
                for movie_id in recommendations.index:
                    title = u_item[u_item["movie_id"] == movie_id]["title"].values[0]
                    movie_data = get_movie_data(title)
                    poster_url = movie_data.get("Poster")
                    if poster_url and poster_url != "N/A":
                        st.image(poster_url, width=100)
                    st.write(f"**{title}** - Predicted Rating: {recommendations[movie_id]:.2f}")
            else:
                st.warning("No similar users with enough data.")
    else:
        top_movies = u_data.groupby("item_id")["rating"].mean().sort_values(ascending=False)
        top_unseen = top_movies[~seen_movies].head(num_recs)

        st.write("### Recommended Popular Movies:")
        for movie_id in top_unseen.index:
            title = u_item[u_item["movie_id"] == movie_id]["title"].values[0]
            movie_data = get_movie_data(title)
            poster_url = movie_data.get("Poster")
            if poster_url and poster_url != "N/A":
                st.image(poster_url, width=100)
            st.write(f"**{title}** - Avg Rating: {top_unseen[movie_id]:.2f}")

# --- Sidebar Data Exploration ---
st.sidebar.title("\U0001F4CA Data Exploration")
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
