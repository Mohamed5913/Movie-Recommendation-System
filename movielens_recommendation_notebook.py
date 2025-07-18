import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("📊 MovieLens 100K Movie Recommendation System")
print("=" * 50)
print("Loading and preprocessing dataset...")

class MovieLensDataLoader:
    """Class to handle MovieLens 100K dataset loading and preprocessing"""
    
    def __init__(self, ratings_file='ml-100k/u.data', movies_file='ml-100k/u.item', users_file='ml-100k/u.user', genre_file='ml-100k/u.genre'):
        self.ratings_file = ratings_file
        self.movies_file = movies_file
        self.users_file = users_file
        self.genre_file = genre_file
        self.ratings = None
        self.movies = None
        self.users = None
        self.genres = None
        self.user_item_matrix = None
        
    def load_data(self):
        """Load ratings, movies, users and genres data"""
        try:
            # Load ratings data (u.data): user_id | item_id | rating | timestamp
            self.ratings = pd.read_csv(
                self.ratings_file, 
                sep='\t', 
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                engine='python'
            )
            
            # Load movies data (u.item): movie_id | title | release_date | video_release_date |
            # IMDb URL | unknown | Action | Adventure | Animation | ... (19 genres)
            movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 
                           'imdb_url']
            
            # Add genre columns
            with open(self.genre_file, 'r') as f:
                genres = [line.split('|')[0] for line in f.readlines() if line.strip()]
                movie_columns.extend(genres)
            
            self.movies = pd.read_csv(
                self.movies_file, 
                sep='|', 
                names=movie_columns,
                engine='python',
                encoding='latin-1'
            )
            
            # Load user data (u.user): user_id | age | gender | occupation | zip_code
            self.users = pd.read_csv(
                self.users_file,
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
            
            # Load genre data
            self.genres = pd.read_csv(
                self.genre_file,
                sep='|',
                names=['genre', 'genre_id'],
                engine='python'
            )
            
            print(f"✅ Loaded {len(self.ratings)} ratings, {len(self.movies)} movies, and {len(self.users)} users")
            return True
            
        except FileNotFoundError:
            print("❌ Dataset files not found. Please make sure all MovieLens 100K dataset files are in the ml-100k folder.")
            print("📥 Download from: https://grouplens.org/datasets/movielens/100k/")
            return False
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        if self.ratings is None:
            print("❌ Ratings data not loaded. Please load data first.")
            return None
            
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        print(f"📊 User-item matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def get_dataset_info(self):
        """Display dataset statistics"""
        if self.ratings is None or self.movies is None or self.users is None:
            print("❌ Data not loaded. Please load data first.")
            return
            
        print("\n📈 Dataset Statistics:")
        print(f"• Users: {self.ratings['user_id'].nunique()}")
        print(f"• Movies: {self.ratings['movie_id'].nunique()}")
        print(f"• Ratings: {len(self.ratings)}")
        print(f"• Rating scale: {self.ratings['rating'].min()} - {self.ratings['rating'].max()}")
        print(f"• Average rating: {self.ratings['rating'].mean():.2f}")
        print(f"• Matrix sparsity: {(1 - len(self.ratings) / (self.ratings['user_id'].nunique() * self.ratings['movie_id'].nunique())) * 100:.1f}%")
        
        # User demographics
        print(f"\n👥 User Demographics:")
        print(f"• Gender: {self.users['gender'].value_counts().to_dict()}")
        print(f"• Age range: {self.users['age'].min()} - {self.users['age'].max()} (avg: {self.users['age'].mean():.1f})")
        print(f"• Top 5 occupations: {self.users['occupation'].value_counts().head(5).to_dict()}")
        
        # Rating distribution
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        self.ratings['rating'].hist(bins=5, alpha=0.7, color='skyblue')
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        ratings_per_user = self.ratings.groupby('user_id').size()
        ratings_per_user.hist(bins=20, alpha=0.7, color='lightgreen')
        plt.title('Ratings per User Distribution')
        plt.xlabel('Number of Ratings')
        plt.ylabel('Number of Users')
        
        plt.tight_layout()
        plt.show()
        
        # Movie genre distribution
        genre_cols = self.genres['genre'].tolist()
        genre_counts = {genre: self.movies[genre].sum() for genre in genre_cols}
        genre_df = pd.DataFrame({'genre': list(genre_counts.keys()), 
                                'count': list(genre_counts.values())})
        genre_df = genre_df.sort_values('count', ascending=False)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x='genre', y='count', data=genre_df)
        plt.title('Movie Genre Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Initialize and load data
loader = MovieLensDataLoader()
loader.load_data()
user_item_matrix = loader.create_user_item_matrix()
loader.get_dataset_info()

# ==============================================================================
# 2. RECOMMENDATION SYSTEM CLASSES
# ==============================================================================

class UserBasedCollaborativeFiltering:
    """User-based Collaborative Filtering implementation"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_similarity = None
        
    def compute_user_similarity(self, metric='pearson'):
        """Compute user similarity matrix"""
        print(f"\n🔄 Computing user similarity using {metric} correlation...")
        
        if metric == 'pearson':
            # Pearson correlation coefficient
            self.user_similarity = self.user_item_matrix.T.corr()
        elif metric == 'cosine':
            # Cosine similarity
            user_matrix = self.user_item_matrix.values
            similarity = cosine_similarity(user_matrix)
            self.user_similarity = pd.DataFrame(
                similarity, 
                index=self.user_item_matrix.index, 
                columns=self.user_item_matrix.index
            )
        
        # Fill NaN values with 0
        self.user_similarity = self.user_similarity.fillna(0)
        
        print(f"✅ User similarity matrix computed: {self.user_similarity.shape}")
        return self.user_similarity
    
    def predict_rating(self, user_id, movie_id, k=10):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_similarity.index:
            return 0
        
        # Get k most similar users
        similar_users = self.user_similarity[user_id].abs().sort_values(ascending=False)[1:k+1]
        
        # Calculate weighted average
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user, similarity in similar_users.items():
            if self.user_item_matrix.loc[similar_user, movie_id] > 0:
                weighted_sum += similarity * self.user_item_matrix.loc[similar_user, movie_id]
                similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0
        
        return weighted_sum / similarity_sum
    
    def recommend_movies(self, user_id, n_recommendations=10, k=10):
        """Generate movie recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies not rated by the user
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id, k)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

class ItemBasedCollaborativeFiltering:
    """Item-based Collaborative Filtering implementation"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.item_similarity = None
        
    def compute_item_similarity(self, metric='pearson'):
        """Compute item similarity matrix"""
        print(f"\n🔄 Computing item similarity using {metric} correlation...")
        
        if metric == 'pearson':
            # Pearson correlation coefficient
            self.item_similarity = self.user_item_matrix.corr()
        elif metric == 'cosine':
            # Cosine similarity
            item_matrix = self.user_item_matrix.T.values
            similarity = cosine_similarity(item_matrix)
            self.item_similarity = pd.DataFrame(
                similarity, 
                index=self.user_item_matrix.columns, 
                columns=self.user_item_matrix.columns
            )
        
        # Fill NaN values with 0
        self.item_similarity = self.item_similarity.fillna(0)
        
        print(f"✅ Item similarity matrix computed: {self.item_similarity.shape}")
        return self.item_similarity
    
    def predict_rating(self, user_id, movie_id, k=10):
        """Predict rating for a user-movie pair"""
        if user_id not in self.user_item_matrix.index or movie_id not in self.item_similarity.index:
            return 0
        
        # Get k most similar items that the user has rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return 0
        
        # Get similarities with rated items
        similarities = self.item_similarity[movie_id][rated_items].abs().sort_values(ascending=False)[:k]
        
        # Calculate weighted average
        weighted_sum = 0
        similarity_sum = 0
        
        for item, similarity in similarities.items():
            weighted_sum += similarity * user_ratings[item]
            similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 0
        
        return weighted_sum / similarity_sum
    
    def recommend_movies(self, user_id, n_recommendations=10, k=10):
        """Generate movie recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies not rated by the user
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id, k)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

class SVDMatrixFactorization:
    """SVD Matrix Factorization for recommendation"""
    
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.svd_model = None
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, n_components=50):
        """Fit SVD model"""
        print(f"\n🔄 Training SVD model with {n_components} components...")
        
        # Create sparse matrix (only non-zero entries)
        sparse_matrix = csr_matrix(self.user_item_matrix.values)
        
        # Fit SVD model
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(sparse_matrix)
        self.item_factors = self.svd_model.components_
        
        print(f"✅ SVD model trained successfully")
        print(f"• User factors shape: {self.user_factors.shape}")
        print(f"• Item factors shape: {self.item_factors.shape}")
        
        return self
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a user-movie pair"""
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
            
            # Compute dot product of user and item factors
            predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[:, movie_idx])
            
            # Clip to valid rating range
            return np.clip(predicted_rating, 1, 5)
        except:
            return 0
    
    def recommend_movies(self, user_id, n_recommendations=10):
        """Generate movie recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies not rated by the user
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating and return top n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# ==============================================================================
# 3. EVALUATION METRICS
# ==============================================================================

class RecommendationEvaluator:
    """Class to evaluate recommendation system performance"""
    
    def __init__(self, user_item_matrix, ratings_df):
        self.user_item_matrix = user_item_matrix
        self.ratings_df = ratings_df
        
    def precision_at_k(self, recommendations, actual_ratings, k=10, threshold=3.5):
        """Calculate Precision@K"""
        if len(recommendations) == 0:
            return 0
        
        # Get top k recommendations
        top_k_recs = [rec[0] for rec in recommendations[:k]]
        
        # Get highly rated items (rating >= threshold) from actual ratings
        highly_rated = set(actual_ratings[actual_ratings >= threshold].index)
        
        # Calculate precision
        relevant_recommended = len(set(top_k_recs) & highly_rated)
        precision = relevant_recommended / min(k, len(top_k_recs))
        
        return precision
    
    def recall_at_k(self, recommendations, actual_ratings, k=10, threshold=3.5):
        """Calculate Recall@K"""
        if len(recommendations) == 0:
            return 0
        
        # Get top k recommendations
        top_k_recs = [rec[0] for rec in recommendations[:k]]
        
        # Get highly rated items (rating >= threshold) from actual ratings
        highly_rated = set(actual_ratings[actual_ratings >= threshold].index)
        
        if len(highly_rated) == 0:
            return 0
        
        # Calculate recall
        relevant_recommended = len(set(top_k_recs) & highly_rated)
        recall = relevant_recommended / len(highly_rated)
        
        return recall
    
    def evaluate_model(self, model, test_users, k=10):
        """Evaluate model performance"""
        precisions = []
        recalls = []
        
        for user_id in test_users:
            if user_id in self.user_item_matrix.index:
                # Get recommendations
                recommendations = model.recommend_movies(user_id, k)
                
                # Get actual ratings
                actual_ratings = self.user_item_matrix.loc[user_id]
                
                # Filter to only include movies the user has rated
                actual_ratings = actual_ratings[actual_ratings > 0]
                
                # Calculate metrics
                precision = self.precision_at_k(recommendations, actual_ratings, k)
                recall = self.recall_at_k(recommendations, actual_ratings, k)
                
                precisions.append(precision)
                recalls.append(recall)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        return avg_precision, avg_recall



print("\n" + "="*50)
print("🚀 TRAINING RECOMMENDATION MODELS")
print("="*50)

# Initialize models
user_cf = UserBasedCollaborativeFiltering(user_item_matrix)
item_cf = ItemBasedCollaborativeFiltering(user_item_matrix)
svd_cf = SVDMatrixFactorization(user_item_matrix)

# Train models
user_cf.compute_user_similarity(metric='pearson')
item_cf.compute_item_similarity(metric='pearson')
svd_cf.fit(n_components=50)

# Select test users
test_users = user_item_matrix.index[:20].tolist()
evaluator = RecommendationEvaluator(user_item_matrix, loader.ratings)

print("\n" + "="*50)
print("📊 MODEL EVALUATION")
print("="*50)

# Evaluate models
models = {
    'User-based CF': user_cf,
    'Item-based CF': item_cf,
    'SVD Matrix Factorization': svd_cf
}

results = {}
for name, model in models.items():
    print(f"\n🔍 Evaluating {name}...")
    precision, recall = evaluator.evaluate_model(model, test_users, k=10)
    results[name] = {'precision': precision, 'recall': recall}
    print(f"• Precision@10: {precision:.4f}")
    print(f"• Recall@10: {recall:.4f}")

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
models_names = list(results.keys())
precisions = [results[model]['precision'] for model in models_names]
recalls = [results[model]['recall'] for model in models_names]

x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, precisions, width, label='Precision@10', alpha=0.8)
plt.bar(x + width/2, recalls, width, label='Recall@10', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, models_names, rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
# F1 scores
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
plt.bar(models_names, f1_scores, alpha=0.8, color='orange')
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("🎬 INTERACTIVE MOVIE RECOMMENDATIONS")
print("="*50)

def display_recommendations(user_id, model_name, model, n_recs=10):
    """Display recommendations for a user"""
    print(f"\n👤 User {user_id} - {model_name} Recommendations:")
    print("-" * 50)
    
    # Get user's rated movies
    user_ratings = user_item_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].sort_values(ascending=False)
    
    print(f"📽️  User's Top Rated Movies:")
    for i, (movie_id, rating) in enumerate(rated_movies.head(5).items()):
        movie_row = loader.movies[loader.movies['movie_id'] == movie_id]
        if not movie_row.empty:
            movie_title = movie_row['title'].iloc[0]
            print(f"  {i+1}. {movie_title} (Rating: {rating})")
    
    # Get recommendations
    recommendations = model.recommend_movies(user_id, n_recs)
    
    print(f"\n🎯 Top {n_recs} Recommendations:")
    for i, (movie_id, pred_rating) in enumerate(recommendations):
        movie_row = loader.movies[loader.movies['movie_id'] == movie_id]
        if not movie_row.empty:
            movie_title = movie_row['title'].iloc[0]
            print(f"  {i+1}. {movie_title} (Predicted Rating: {pred_rating:.2f})")

# Demo for different users and models
demo_users = [1, 5, 10, 15, 20]
demo_user = demo_users[0]

print(f"\n🎭 Demonstration for User {demo_user}:")

for model_name, model in models.items():
    display_recommendations(demo_user, model_name, model, n_recs=5)



print("\n" + "="*50)
print("📈 ADVANCED ANALYSIS")
print("="*50)

# Analyze recommendation diversity
def calculate_diversity(recommendations_list):
    """Calculate diversity of recommendations"""
    all_recs = [rec[0] for recs in recommendations_list for rec in recs]
    unique_recs = set(all_recs)
    return len(unique_recs) / len(all_recs) if all_recs else 0

# Analyze model diversity
print("\n🔍 Recommendation Diversity Analysis:")
for model_name, model in models.items():
    all_recommendations = []
    for user_id in test_users[:10]:
        if user_id in user_item_matrix.index:
            recs = model.recommend_movies(user_id, 10)
            all_recommendations.append(recs)
    
    diversity = calculate_diversity(all_recommendations)
    print(f"• {model_name}: {diversity:.4f}")

# Analyze popular vs niche recommendations
print("\n📊 Popular vs Niche Movie Analysis:")
movie_popularity = loader.ratings.groupby('movie_id').size()
popular_movies = set(movie_popularity.nlargest(50).index)

for model_name, model in models.items():
    popular_recs = 0
    total_recs = 0
    
    for user_id in test_users[:10]:
        if user_id in user_item_matrix.index:
            recs = model.recommend_movies(user_id, 10)
            for rec, _ in recs:
                total_recs += 1
                if rec in popular_movies:
                    popular_recs += 1
    
    popularity_ratio = popular_recs / total_recs if total_recs > 0 else 0
    print(f"• {model_name}: {popularity_ratio:.2%} popular movies")



print("\n" + "="*50)
print("📋 SUMMARY AND CONCLUSIONS")
print("="*50)

print("\n✅ Implementation Complete!")
print("🎯 Features Implemented:")
print("  • User-based Collaborative Filtering with Pearson correlation")
print("  • Item-based Collaborative Filtering (Bonus)")
print("  • SVD Matrix Factorization (Bonus)")
print("  • Precision@K and Recall@K evaluation")
print("  • Comprehensive performance analysis")
print("  • Interactive recommendation demo")

print("\n📊 Key Findings:")
best_model = max(results.keys(), key=lambda x: results[x]['precision'])
print(f"  • Best performing model: {best_model}")
print(f"  • Best Precision@10: {results[best_model]['precision']:.4f}")
print(f"  • Dataset: MovieLens 100K with {len(loader.ratings)} ratings from {loader.ratings['user_id'].nunique()} users on {loader.ratings['movie_id'].nunique()} movies")
print(f"  • Matrix sparsity: {(1 - len(loader.ratings) / (loader.ratings['user_id'].nunique() * loader.ratings['movie_id'].nunique())) * 100:.1f}%")
print(f"  • Rating scale: {loader.ratings['rating'].min()} - {loader.ratings['rating'].max()} (avg: {loader.ratings['rating'].mean():.2f})")

print("\n🔮 Recommendations for Improvement:")
print("  • Use more sophisticated similarity measures (e.g., adjusted cosine)")
print("  • Implement deep learning approaches (Neural Collaborative Filtering)")
print("  • Add content-based features (genres, actors, directors)")
print("  • Implement hybrid recommendation systems")
print("  • Use cross-validation for more robust evaluation")

print("\n" + "="*50)
print("🎉 ANALYSIS COMPLETE!")
print("="*50)


# ==============================================================================
# 6. GUI RECOMMENDATION SYSTEM
# ==============================================================================

def gui_recommendation_system(loader, models):
    """GUI-based interactive movie recommendation system using Tkinter"""
    
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext
    import threading
    
    # Initialize user ratings dictionary
    user_ratings = {}
    temp_user_id = 'temp_user'
    
    # Create the main window
    root = tk.Tk()
    root.title("MovieLens Recommendation System")
    root.geometry("800x600")
    root.configure(bg="#f5f5f5")
    
    # Set styles
    style = ttk.Style()
    style.configure("TFrame", background="#f5f5f5")
    style.configure("TButton", background="#4CAF50", font=('Arial', 10))
    style.configure("TLabel", background="#f5f5f5", font=('Arial', 11))
    style.configure("Header.TLabel", background="#f5f5f5", font=('Arial', 14, 'bold'))
    
    # Create a notebook (tabbed interface)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Create tabs for each function
    search_tab = ttk.Frame(notebook)
    recommend_tab = ttk.Frame(notebook)
    rated_tab = ttk.Frame(notebook)
    
    notebook.add(search_tab, text="Search & Rate Movies")
    notebook.add(recommend_tab, text="Get Recommendations")
    notebook.add(rated_tab, text="Your Rated Movies")
    
    # ====== Search & Rate Movies Tab ======
    search_header = ttk.Label(search_tab, text="Search for Movies to Rate", style="Header.TLabel")
    search_header.pack(pady=10)
    
    search_frame = ttk.Frame(search_tab)
    search_frame.pack(fill='x', padx=20)
    
    search_label = ttk.Label(search_frame, text="Enter movie title:")
    search_label.pack(side='left', padx=5)
    
    search_entry = ttk.Entry(search_frame, width=40)
    search_entry.pack(side='left', padx=5)
    
    # Results frame with scrollable list
    results_frame = ttk.LabelFrame(search_tab, text="Search Results")
    results_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    # Scrollable list for search results
    search_results = ttk.Treeview(results_frame, columns=('id', 'title'), show='headings', height=10)
    search_results.heading('id', text='ID')
    search_results.heading('title', text='Movie Title')
    search_results.column('id', width=50)
    search_results.column('title', width=400)
    search_results.pack(side='left', fill='both', expand=True)
    
    results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=search_results.yview)
    results_scroll.pack(side='right', fill='y')
    search_results.configure(yscrollcommand=results_scroll.set)
    
    # Rating frame
    rating_frame = ttk.Frame(search_tab)
    rating_frame.pack(fill='x', padx=20, pady=10)
    
    rating_label = ttk.Label(rating_frame, text="Rate selected movie (1-5):")
    rating_label.pack(side='left', padx=5)
    
    rating_var = tk.StringVar()
    rating_combobox = ttk.Combobox(rating_frame, textvariable=rating_var, width=5, state="readonly")
    rating_combobox['values'] = ("1", "2", "3", "4", "5")
    rating_combobox.pack(side='left', padx=5)
    
    # ====== Get Recommendations Tab ======
    recommend_header = ttk.Label(recommend_tab, text="Get Movie Recommendations", style="Header.TLabel")
    recommend_header.pack(pady=10)
    
    model_frame = ttk.Frame(recommend_tab)
    model_frame.pack(fill='x', padx=20, pady=10)
    
    model_label = ttk.Label(model_frame, text="Select recommendation model:")
    model_label.pack(side='left', padx=5)
    
    model_var = tk.StringVar()
    model_combobox = ttk.Combobox(model_frame, textvariable=model_var, width=30, state="readonly")
    model_combobox['values'] = tuple(models.keys())
    model_combobox.pack(side='left', padx=5)
    model_combobox.current(0)  # Set default to first model
    
    # Results frame for recommendations
    rec_results_frame = ttk.LabelFrame(recommend_tab, text="Recommendations")
    rec_results_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    # Text widget to display recommendations
    rec_text = scrolledtext.ScrolledText(rec_results_frame, wrap=tk.WORD, width=70, height=15)
    rec_text.pack(fill='both', expand=True, padx=5, pady=5)
    rec_text.config(state='disabled')
    
    # ====== Your Rated Movies Tab ======
    rated_header = ttk.Label(rated_tab, text="Your Rated Movies", style="Header.TLabel")
    rated_header.pack(pady=10)
    
    # Rated movies list
    rated_frame = ttk.Frame(rated_tab)
    rated_frame.pack(fill='both', expand=True, padx=20, pady=10)
    
    rated_movies_list = ttk.Treeview(rated_frame, columns=('title', 'rating'), show='headings', height=15)
    rated_movies_list.heading('title', text='Movie Title')
    rated_movies_list.heading('rating', text='Your Rating')
    rated_movies_list.column('title', width=400)
    rated_movies_list.column('rating', width=100)
    rated_movies_list.pack(side='left', fill='both', expand=True)
    
    rated_scroll = ttk.Scrollbar(rated_frame, orient="vertical", command=rated_movies_list.yview)
    rated_scroll.pack(side='right', fill='y')
    rated_movies_list.configure(yscrollcommand=rated_scroll.set)
    
    # Function to update the rated movies tab
    def update_rated_movies_tab():
        # Clear existing items
        for item in rated_movies_list.get_children():
            rated_movies_list.delete(item)
            
        # Add all rated movies
        for movie_id, rating in user_ratings.items():
            try:
                movie_row = loader.movies[loader.movies['movie_id'] == movie_id]
                if not movie_row.empty:
                    movie_title = movie_row['title'].iloc[0]
                    rated_movies_list.insert('', 'end', values=(movie_title, f"{rating}/5"))
            except Exception as e:
                print(f"Error displaying rated movie {movie_id}: {e}")
    
    # Function to search for movies
    def search_movies():
        search_query = search_entry.get().strip()
        if not search_query:
            messagebox.showwarning("Input Required", "Please enter a movie title to search.")
            return
            
        # Clear existing results
        for item in search_results.get_children():
            search_results.delete(item)
            
        try:
            # Search for movies
            matching_movies = loader.movies[loader.movies['title'].str.contains(search_query, case=False)]
            
            if len(matching_movies) == 0:
                messagebox.showinfo("No Results", "No movies found matching your query.")
                return
                
            # Display search results
            for _, movie in matching_movies.iterrows():
                search_results.insert('', 'end', values=(movie['movie_id'], movie['title']))
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during search: {str(e)}")
    
    # Function to rate a movie
    def rate_movie():
        selection = search_results.selection()
        if not selection:
            messagebox.showwarning("Selection Required", "Please select a movie from the search results.")
            return
            
        if not rating_var.get():
            messagebox.showwarning("Rating Required", "Please select a rating (1-5).")
            return
            
        try:
            # Get selected movie
            movie_id = search_results.item(selection[0])['values'][0]
            movie_title = search_results.item(selection[0])['values'][1]
            rating = float(rating_var.get())
            
            # Add to user ratings
            user_ratings[movie_id] = rating
            messagebox.showinfo("Success", f"Added rating: '{movie_title}' - {rating}/5")
            
            # Update rated movies tab
            update_rated_movies_tab()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    # Function to generate recommendations
    def get_recommendations():
        if len(user_ratings) == 0:
            messagebox.showwarning("No Ratings", "You haven't rated any movies yet. Please rate some movies first.")
            return
            
        model_name = model_var.get()
        if not model_name:
            messagebox.showwarning("Selection Required", "Please select a recommendation model.")
            return
            
        try:
            # Show loading message
            rec_text.config(state='normal')
            rec_text.delete(1.0, tk.END)
            rec_text.insert(tk.END, "Generating recommendations, please wait...")
            rec_text.config(state='disabled')
            root.update()
            
            # Run recommendations in a thread to avoid freezing UI
            def run_recommendations():
                try:
                    # Create a temporary row in the user-item matrix
                    all_movie_ids = loader.user_item_matrix.columns
                    
                    temp_user_ratings = pd.Series(0, index=all_movie_ids)
                    for movie_id, rating in user_ratings.items():
                        if movie_id in all_movie_ids:
                            temp_user_ratings[movie_id] = rating
                            
                    model = models[model_name]
                    
                    # Generate recommendations based on model type
                    if model_name == 'User-based CF':
                        # Add temporary user to matrix
                        temp_matrix = loader.user_item_matrix.copy()
                        temp_matrix.loc[temp_user_id] = temp_user_ratings
                        
                        # Recompute similarity for this user
                        model.user_item_matrix = temp_matrix
                        model.compute_user_similarity()
                        
                        # Get recommendations
                        recommendations = model.recommend_movies(temp_user_id, n_recommendations=10)
                    
                    elif model_name == 'Item-based CF':
                        # Use existing item similarities but new user vector
                        temp_df = pd.DataFrame([temp_user_ratings])
                        temp_df.index = [temp_user_id]
                        model.user_item_matrix = pd.concat([loader.user_item_matrix, temp_df])
                        recommendations = model.recommend_movies(temp_user_id, n_recommendations=10)
                    
                    else:  # SVD model
                        # For SVD, we'll use the item-based approach as a fallback
                        item_cf = models['Item-based CF']
                        temp_df = pd.DataFrame([temp_user_ratings])
                        temp_df.index = [temp_user_id]
                        item_cf.user_item_matrix = pd.concat([loader.user_item_matrix, temp_df])
                        recommendations = item_cf.recommend_movies(temp_user_id, n_recommendations=10)
                    
                    # Display recommendations
                    rec_text.config(state='normal')
                    rec_text.delete(1.0, tk.END)
                    rec_text.insert(tk.END, f"Top 10 Movie Recommendations ({model_name}):\n\n")
                    
                    for i, (movie_id, pred_rating) in enumerate(recommendations):
                        try:
                            movie_row = loader.movies[loader.movies['movie_id'] == movie_id]
                            if not movie_row.empty:
                                movie_title = movie_row['title'].iloc[0]
                                rec_text.insert(tk.END, f"{i+1}. {movie_title} (Predicted Rating: {pred_rating:.2f})\n")
                        except Exception as e:
                            rec_text.insert(tk.END, f"{i+1}. Movie ID {movie_id} (Error retrieving title)\n")
                            
                    rec_text.config(state='disabled')
                    
                except Exception as e:
                    # Show error in the text box
                    rec_text.config(state='normal')
                    rec_text.delete(1.0, tk.END)
                    rec_text.insert(tk.END, f"Error generating recommendations: {str(e)}")
                    rec_text.config(state='disabled')
            
            # Start recommendation thread
            threading.Thread(target=run_recommendations).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    # Add buttons for actions
    search_button = ttk.Button(search_frame, text="Search", command=search_movies)
    search_button.pack(side='left', padx=5)
    
    rate_button = ttk.Button(rating_frame, text="Rate Movie", command=rate_movie)
    rate_button.pack(side='left', padx=5)
    
    recommend_button = ttk.Button(model_frame, text="Get Recommendations", command=get_recommendations)
    recommend_button.pack(side='left', padx=5)
    
    # Bind the Enter key to search
    search_entry.bind('<Return>', lambda event: search_movies())
    
    # Status bar at the bottom
    status_var = tk.StringVar()
    status_var.set("Ready - MovieLens 100K Recommendation System")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Start the GUI
    root.mainloop()

print("\n" + "="*50)
print("🎮 STARTING GUI RECOMMENDATION SYSTEM")
print("="*50)
print("Please wait while the GUI loads...")

# Run the GUI recommendation system instead of the terminal-based one
gui_recommendation_system(loader, models)