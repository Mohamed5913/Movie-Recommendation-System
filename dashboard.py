import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load environment variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Load MovieLens 100K data
column_names = ['userId', 'movieId', 'rating', 'timestamp']
data = pd.read_csv("ml-100k/u.data", sep='\t', names=column_names)

item_columns = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', names=item_columns)

# Load genres and remove 'unknown'
genre_df = pd.read_csv("ml-100k/u.genre", sep='|', names=['genre', 'genre_id'], encoding='latin-1')
genres = genre_df['genre'].dropna().tolist()
genres = [g for g in genres if g.lower() != 'unknown']

# Create user-item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Cache similarity matrices
@st.cache_data(show_spinner=False)
def compute_user_similarity():
    user_similarity = cosine_similarity(user_movie_matrix)
    return pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

@st.cache_data(show_spinner=False)
def compute_item_similarity():
    item_similarity = cosine_similarity(user_movie_matrix.T)
    return pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

@st.cache_data(show_spinner=False)
def compute_svd_similarity():
    svd = TruncatedSVD(n_components=20)
    svd_matrix = svd.fit_transform(user_movie_matrix)
    svd_sim = cosine_similarity(svd_matrix)
    return pd.DataFrame(svd_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def fetch_poster(imdb_url):
    try:
        imdb_id = imdb_url.strip().split('/')[-2]
        api_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        response = requests.get(api_url)
        data = response.json()
        return data.get("Poster", "")
    except Exception:
        return ""

user_similarity_df = compute_user_similarity()
item_similarity_df = compute_item_similarity()
svd_similarity_df = compute_svd_similarity()

def get_top_n_recommendations(user_id, similarity_df, n=5, genre_filter=None):
    sim_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    user_ratings = user_movie_matrix.loc[user_id]
    weighted_ratings = pd.Series(dtype=float)
    for other_user, sim_score in sim_users.items():
        other_ratings = user_movie_matrix.loc[other_user]
        for movie_id, rating in other_ratings.items():
            if user_ratings[movie_id] == 0:
                if movie_id not in weighted_ratings:
                    weighted_ratings[movie_id] = 0
                weighted_ratings[movie_id] += rating * sim_score
    recommendations = weighted_ratings.sort_values(ascending=False)
    movie_info = movies[movies['movieId'].isin(recommendations.index)][['movieId', 'title', 'IMDb_URL']].copy()

    if genre_filter:
        genre_cols = [g for g in genre_filter if g in movies.columns]
        genre_mask = movies.set_index('movieId').loc[movie_info['movieId']][genre_cols].sum(axis=1) > 0
        movie_info = movie_info[genre_mask.values]

    return movie_info.head(n)

# Streamlit UI
st.title("🎬 Movie Recommendation System")

model = st.selectbox("Select Recommendation Model:", ["User-Based CF", "Item-Based CF", "SVD-Based CF"])

# Limit default users to 5 and allow dynamic addition
default_users = user_movie_matrix.index[:5].tolist()
default_user_names = {uid: f"User {uid}" for uid in default_users}

if 'custom_users' not in st.session_state:
    st.session_state.custom_users = {}

custom_user_ids = list(st.session_state.custom_users.keys())
custom_user_names = {uid: name for uid, name in st.session_state.custom_users.items()}

all_users = default_users + custom_user_ids
all_user_names = {**default_user_names, **custom_user_names}

selected_user_id = st.selectbox("Select User Account:", options=all_users, format_func=lambda x: all_user_names[x])
user_id = selected_user_id

# Option to add new user
st.markdown("---")
st.subheader("➕ Add New User")

new_user_name = st.text_input("Enter new user name:")
selected_genres = st.multiselect("Filter movies by genre:", genres)
filtered_movies = movies.copy()
if selected_genres:
    genre_cols = [g for g in selected_genres if g in movies.columns]
    filtered_movies = filtered_movies[filtered_movies[genre_cols].sum(axis=1) > 0]

search_movie = st.selectbox("Search and rate a movie:", filtered_movies['title'].sort_values().tolist())
rating = st.slider("Your Rating:", 1, 5, 3)

if st.button("Add User"):
    if new_user_name.strip():
        new_id = user_movie_matrix.index.max() + 1
        new_row = pd.Series(0, index=user_movie_matrix.columns)
        movie_id = filtered_movies[filtered_movies['title'] == search_movie]['movieId'].values[0]
        new_row[movie_id] = rating
        user_movie_matrix.loc[new_id] = new_row
        st.session_state.custom_users[new_id] = new_user_name.strip()
        st.success(f"New user '{new_user_name}' added with ID {new_id}")
        st.rerun()
    else:
        st.warning("Please enter a valid user name.")

genre_filter = st.multiselect("Select preferred genres (optional):", genres)
top_n = st.slider("Number of Recommendations:", 1, 20, 5)

if st.button("Get Recommendations"):
    if model == "User-Based CF":
        recs = get_top_n_recommendations(user_id, user_similarity_df, n=top_n, genre_filter=genre_filter)
    elif model == "Item-Based CF":
        recs = get_top_n_recommendations(user_id, item_similarity_df, n=top_n, genre_filter=genre_filter)
    else:
        recs = get_top_n_recommendations(user_id, svd_similarity_df, n=top_n, genre_filter=genre_filter)

    st.write(f"Top {top_n} recommendations for {all_user_names[user_id]}:")

    for _, row in recs.iterrows():
        poster_url = fetch_poster(row['IMDb_URL'])
        cols = st.columns([1, 4])
        with cols[0]:
            if poster_url:
                st.image(poster_url, use_column_width=True)
        with cols[1]:
            st.markdown(f"**{row['title']}**")
            st.markdown(f"[🎬 IMDb Page]({row['IMDb_URL']})")
        st.markdown("---")

    # Genre distribution visualization
    genre_data = movies.set_index('movieId').loc[recs['movieId']][genres]
    genre_counts = genre_data.sum().sort_values(ascending=False)
    st.subheader("🎨 Genre Distribution of Recommendations")
    fig, ax = plt.subplots()
    genre_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # Top-rated genres by user
    st.subheader("📊 Top-Rated Genres by This User")
    user_ratings = data[data['userId'] == user_id]
    rated_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]
    genre_totals = rated_movies[genres].sum()
    genre_avg_ratings = rated_movies[genres].multiply(user_ratings.set_index('movieId')['rating'], axis=0).sum() / genre_totals
    genre_avg_ratings = genre_avg_ratings.dropna().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    genre_avg_ratings.plot(kind='bar', ax=ax2, color='orange')
    st.pyplot(fig2)

st.sidebar.title("🎭 Genres Available")
st.sidebar.write(", ".join(genres))
