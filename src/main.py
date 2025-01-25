from src.data_loader import load_data
from src.matrix_factorization import perform_svd
from src.recommender import recommend_movies

if __name__ == "__main__":
    print("Loading data...")
    data, movies = load_data()

    print("Performing matrix factorization...")
    svd_matrix = perform_svd(data.values)

    user_id = 1  # Example user
    print(f"Recommended movies for User {user_id}:")
    recommendations = recommend_movies(user_id, svd_matrix, movies)
    print(recommendations)
