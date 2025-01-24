import numpy as np

def recommend_movies(user_id, reconstructed_matrix, movies_df, top_n=5):
    user_ratings = reconstructed_matrix[user_id]
    recommended_movie_indices = np.argsort(user_ratings)[::-1][:top_n]
    recommended_movies = movies_df.iloc[recommended_movie_indices]
    return recommended_movies[['movieId', 'title']]

if __name__ == "__main__":
    from data_loader import load_data
    from matrix_factorization import perform_svd
    data, movies = load_data()
    svd_matrix = perform_svd(data.values)
    recommendations = recommend_movies(0, svd_matrix, movies)
    print(recommendations)
