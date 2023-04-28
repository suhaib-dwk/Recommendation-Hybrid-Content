import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('data.csv')

# Preprocessing
df['user_id'] = df['user_id'].astype('category').cat.codes
df['item_id'] = df['item_id'].astype('category').cat.codes

# Collaborative filtering
user_item_matrix = df.pivot_table(
    index='user_id', columns='item_id', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english',
                        analyzer='word', ngram_range=(1, 2))
product_desc = df.set_index('item_id')['product_desc_ar']
product_desc_tfidf = tfidf.fit_transform(product_desc)

# Demographic filtering
pca = PCA(n_components=2)
user_demographics = df[['age', 'gender']]
user_demographics_pca = pca.fit_transform(user_demographics)

# Hybrid system


def recommend(user_id, item_id, method):
    if method == 'collaborative':
        # Get top similar users
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_item_matrix)
        distances, indices = model.kneighbors(
            user_item_matrix.iloc[user_id, :].values.reshape(1, -1), n_neighbors=6)
        similar_users = indices[0][1:]
        # Get top rated items
        top_rated = user_item_matrix.iloc[similar_users, :].mean(
        ).sort_values(ascending=False)
        return top_rated.index[:5]
    elif method == 'content':
        # Get top similar items
        cosine_similarities = cosine_similarity(
            product_desc_tfidf[item_id], product_desc_tfidf).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        return related_docs_indices
    elif method == 'demographic':
        # Get top similar users
        model = NearestNeighbors(metric='euclidean', algorithm='brute')
        model.fit(user_demographics_pca)
        distances, indices = model.kneighbors(
            user_demographics_pca[user_id, :].reshape(1, -1), n_neighbors=6)
        similar_users = indices[0][1:]
        # Get top rated items
        top_rated = user_item_matrix.iloc[similar_users, :].mean(
        ).sort_values(ascending=False)
        return top_rated.index[:5]
