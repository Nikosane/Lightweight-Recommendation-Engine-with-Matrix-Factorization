import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(matrix, title="Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap="viridis")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    from data_loader import load_data
    from matrix_factorization import perform_svd
    data, _ = load_data()
    svd_matrix = perform_svd(data.values)
    plot_heatmap(svd_matrix, "Reconstructed Ratings")
