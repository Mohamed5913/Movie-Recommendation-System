# 🎬 MovieLens Recommendation Dashboard

🌐 Try it live: [Movie Recommendation System on Streamlit](https://movie-recommendation-system-695fvsct77wj48y7qe7x5o.streamlit.app/#movie-recommendation-system)

A dynamic Streamlit-based movie recommendation system powered by collaborative filtering and interactive user input. Built using the MovieLens 100k dataset.

---

## 🚀 Features

- ✅ **User-based Collaborative Filtering** for personalized movie suggestions.
- 👤 **User Management**
  - Start with 5 users (User 1 has real data, others are placeholders).
  - Add new users dynamically.
- ⭐ **Rate Movies**
  - Search by **title**, filter by **genres** and **release year**.
  - Assign ratings (1–5) to any movie.
- 🎯 **Get Recommendations**
  - Select the number of movie recommendations (5–20).
  - See predicted ratings and movie posters.
- 📊 **Data Exploration Tools** (via sidebar)
  - View raw datasets.
  - Plot ratings distribution.
  - Discover top-rated movies.

---

## 📁 Dataset

This app uses the [MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/):

- `u.data` — user ratings
- `u.item` — movie metadata
- `u.user` — user demographics
- `u.genre` — genre definitions

---

## 🔧 Tech Stack

- **Python**
- **Streamlit**
- **Pandas**, **NumPy**
- **Scikit-learn** for cosine similarity
- **Matplotlib**, **Seaborn** for visualizations
- **OMDb API** for movie posters
