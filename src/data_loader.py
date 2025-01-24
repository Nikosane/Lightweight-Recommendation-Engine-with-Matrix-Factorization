import pandas as pd

def load_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
    user_item_matrix.fillna(0, inplace=True)  # Replace NaN with 0
    return user_item_matrix, movies

if __name__ == "__main__":
    data, movies = load_data()
    print(data.head())
