import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Load MovieLens 100K data
column_names = ['userId', 'movieId', 'rating', 'timestamp']
data = pd.read_csv("ml-100k/u.data", sep='\t', names=column_names)

item_columns = ['movieId', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv("ml-100k/u.item", sep='|', encoding='latin-1', names=item_columns)

# Load genres
genre_df = pd.read_csv("ml-100k/u.genre", sep='|', names=['genre', 'genre_id'], encoding='latin-1')
genres = genre_df['genre'].dropna().tolist()

# Create user-item matrix
user_movie_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# Cosine similarity matrices
user_similarity = cosine_similarity(user_movie_matrix)
item_similarity = cosine_similarity(user_movie_matrix.T)

user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
item_similarity_df = pd.DataFrame(item_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

# SVD
svd = TruncatedSVD(n_components=20)
svd_matrix = svd.fit_transform(user_movie_matrix)
svd_sim = cosine_similarity(svd_matrix)
svd_similarity_df = pd.DataFrame(svd_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def get_top_n_recommendations(user_id, similarity_df, n=5, genre_filter=None):
    sim_users = similarity_df[user_id].sort_values(ascending=False)[1:]
    user_ratings = user_movie_matrix.loc[user_id]
    weighted_ratings = pd.Series(dtype=float)
    for other_user, sim_score in sim_users.items():
        other_ratings = user_movie_matrix.loc[other_user]
        for movie_id, rating in other_ratings.items():
            if user_ratings[movie_id] == 0:  # not yet rated
                if movie_id not in weighted_ratings:
                    weighted_ratings[movie_id] = 0
                weighted_ratings[movie_id] += rating * sim_score
    recommendations = weighted_ratings.sort_values(ascending=False)
    movie_info = movies[movies['movieId'].isin(recommendations.index)][['movieId', 'title']].copy()

    if genre_filter:
        genre_cols = [g for g in genre_filter if g in movies.columns]
        genre_mask = movies.set_index('movieId').loc[movie_info['movieId']][genre_cols].sum(axis=1) > 0
        movie_info = movie_info[genre_mask.values]

    return movie_info.head(n)

# Streamlit UI
st.title("🎬 Movie Recommendation System")

model = st.selectbox("Select Recommendation Model:", ["User-Based CF", "Item-Based CF", "SVD-Based CF"])
user_id = st.number_input("Enter User ID:", min_value=int(user_movie_matrix.index.min()), max_value=int(user_movie_matrix.index.max()), value=int(user_movie_matrix.index.min()))
top_n = st.slider("Number of Recommendations:", 1, 20, 5)
genre_filter = st.multiselect("Select preferred genres (optional):", genres)

if st.button("Get Recommendations"):
    if model == "User-Based CF":
        recs = get_top_n_recommendations(user_id, user_similarity_df, n=top_n, genre_filter=genre_filter)
    elif model == "Item-Based CF":
        recs = get_top_n_recommendations(user_id, item_similarity_df, n=top_n, genre_filter=genre_filter)
    else:
        recs = get_top_n_recommendations(user_id, svd_similarity_df, n=top_n, genre_filter=genre_filter)

    st.write(f"Top {top_n} recommendations for User {user_id}:")
    st.table(recs)

st.sidebar.title("🎭 Genres Available")
st.sidebar.write(", ".join(genres))
