import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier


def recommend(data):
    # Define the weights for each component of the recommendation
    CB_weight = 0.7
    CF_weight = 0.1
    DF_weight = 0.2

    # Compute CB_recomender
    search_keywords = data['Search_Keywords'] * 0.1
    favorites = data['Favorites'] * 0.1
    product_ads = data['Product_Advertisement'] * 0.3
    store_names = data['Store_Names'] * 0.05
    main_category = data['Main_Category'] * 0.05
    sub_categories = data['Sub_Categories'] * 0.05
    location_store = data['Location_Store'] * 0.05

    CB_recomender = search_keywords + favorites + product_ads + \
        store_names + main_category + sub_categories + location_store

    # Compute CF_recomender
    CF_recomender = data['Rating'] * 0.1

    # Compute DF_recomender
    age = data['Age'] * 0.05
    gender = data['Gender'] * 0.05
    user_location = data['User_Location'] * 0.05
    sub_categories = data['Sub_Categories'] * 0.05

    DF_recomender = age + gender + user_location + sub_categories

    # Combine the recommendations using the defined weights
    RE = CB_weight * CB_recomender + CF_weight * \
        CF_recomender + DF_weight * DF_recomender

    # Train and fit the KNNWithMeans model
    knn = KNeighborsClassifier()
    knn.fit(RE, data['target'])

    # Compute the cosine similarity between the recommendations
    cos_sim = cosine_similarity(RE)

    # Train and fit the RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(cos_sim, data['target'])

    # Return the top N recommendations based on the trained models
    N = 10
    knn_recs = knn.kneighbors(RE, n_neighbors=N, return_distance=False)
    rfc_recs = rfc.predict(cos_sim)
    combined_recs = (knn_recs + rfc_recs) / 2

    # Save the trained models and their recommendations to Firebase
    # (Code for saving to Firebase not provided in this function)

    return combined_recs
