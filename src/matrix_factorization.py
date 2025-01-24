import numpy as np
from scipy.sparse.linalg import svds

def perform_svd(matrix, k=50):
    U, sigma, Vt = svds(matrix, k=k)
    sigma = np.diag(sigma)
    reconstructed_matrix = np.dot(np.dot(U, sigma), Vt)
    return reconstructed_matrix

if __name__ == "__main__":
    from data_loader import load_data
    data, _ = load_data()
    svd_matrix = perform_svd(data.values)
    print("SVD Matrix shape:", svd_matrix.shape)
