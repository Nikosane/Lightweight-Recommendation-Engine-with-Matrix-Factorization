from sklearn.metrics import mean_squared_error
import numpy as np

def calculate_rmse(original_matrix, reconstructed_matrix):
    mask = original_matrix > 0
    return np.sqrt(mean_squared_error(original_matrix[mask], reconstructed_matrix[mask]))

if __name__ == "__main__":
    from data_loader import load_data
    from matrix_factorization import perform_svd
    data, _ = load_data()
    svd_matrix = perform_svd(data.values)
    print("RMSE:", calculate_rmse(data.values, svd_matrix))
