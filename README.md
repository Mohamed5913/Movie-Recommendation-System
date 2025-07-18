# MovieLens Recommendation System

A comprehensive movie recommendation system built with Python using the MovieLens 100K dataset. This project implements multiple recommendation algorithms and provides an interactive interface for users to rate movies and receive personalized recommendations.

## Dataset

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) which contains:

- 100,000 ratings (1-5) from 943 users on 1,682 movies
- Each user has rated at least 20 movies
- Simple demographic info for users (age, gender, occupation, zip)

The dataset is included in the `ml-100k` directory and consists of several files:

- `u.data`: The main ratings file (user_id, movie_id, rating, timestamp)
- `u.item`: Movie information (title, release date, genres, etc.)
- `u.user`: User demographics
- `u.genre`: Genre definitions

## Features

- **Multiple Recommendation Algorithms**:

  - User-based Collaborative Filtering
  - Item-based Collaborative Filtering
  - SVD Matrix Factorization

- **Comprehensive Evaluation**:

  - Precision@K and Recall@K metrics
  - Model performance comparison
  - Analysis of recommendation diversity
  - Popular vs. niche movie analysis

- **Interactive User Interface**:

  - Search for movies to rate
  - Get personalized recommendations
  - Choose between different recommendation algorithms
  - View your rated movies

- **Data Visualization**:
  - Rating distributions
  - Genre distributions
  - Model performance comparisons

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy

## Installation

1. Clone this repository:

```
git clone <repository-url>
cd movie-recommendation-system
```

2. Install required packages:

```
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

3. Make sure the MovieLens 100K dataset is in the `ml-100k` directory. If not, download it from [here](https://grouplens.org/datasets/movielens/100k/).

## Usage

Run the main script:

```
python movielens_recommendation_notebook.py
```

The script will:

1. Load and preprocess the MovieLens dataset
2. Build and evaluate recommendation models
3. Launch the interactive recommendation system

### Interactive Mode

In interactive mode, you can:

1. **Search for movies to rate**:

   - Enter a movie title (or part of it)
   - Select a movie from the results
   - Rate it on a scale from 1 to 5

2. **Get personalized recommendations**:

   - Select a recommendation algorithm
   - View top 10 movie recommendations based on your ratings

3. **View your rated movies**:
   - See a list of all movies you've rated and your ratings

## Implementation Details

### Data Preprocessing

- Loading and parsing the MovieLens 100K dataset files
- Creating a user-item rating matrix
- Handling missing values and data formatting

### Recommendation Algorithms

1. **User-based Collaborative Filtering**:

   - Computes similarities between users using Pearson correlation
   - Recommends items liked by similar users

2. **Item-based Collaborative Filtering**:

   - Computes similarities between items
   - Recommends items similar to those the user has liked

3. **SVD Matrix Factorization**:
   - Decomposes the user-item matrix using Singular Value Decomposition
   - Uses latent factors to make predictions

### Evaluation Metrics

- **Precision@K**: The proportion of recommended items that are relevant
- **Recall@K**: The proportion of relevant items that are recommended
- **Analysis of recommendation diversity and popularity bias**

## Future Improvements

- Implement hybrid recommendation approaches
- Add content-based filtering using movie genres and descriptions
- Improve the SVD implementation with regularization
- Add support for larger MovieLens datasets (1M, 10M)
- Create a web-based user interface

## License

This project uses the MovieLens dataset which is subject to its own [license terms](https://grouplens.org/datasets/movielens/).

## Acknowledgements

- [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset
- The developers of pandas, numpy, scikit-learn, and other libraries used in this project
