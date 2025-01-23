# Lightweight Recommendation Engine with Matrix Factorization

## Overview
This project implements a lightweight recommendation system using matrix factorization (Singular Value Decomposition - SVD) to provide movie recommendations based on user ratings. The system processes user-item interaction data to predict ratings for unseen movies and generate personalized recommendations.

## Installation

### Prerequisites
Ensure you have Python installed (version 3.x recommended).

### Install Dependencies
Run the following command to install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Project
To execute the recommendation engine, run:
```bash
python main.py
```

## Project Workflow
1. **Data Loading:**
   - Load movie metadata and ratings.
   - Create a user-item interaction matrix.
   
2. **Matrix Factorization:**
   - Decompose the user-item matrix using Singular Value Decomposition (SVD).
   - Reconstruct the matrix to predict missing values.
   
3. **Recommendation Generation:**
   - Generate top-N movie recommendations for a given user.
   
4. **Evaluation:**
   - Calculate RMSE to assess prediction accuracy.

5. **Visualization:**
   - Visualize reconstructed ratings using heatmaps.

## Example Output
```
Loading data...
Performing matrix factorization...
Recommended movies for User 1:
   movieId          title
0       589   Terminator 2
1       783  The Matrix
2       912     Inception
```

## Evaluation
The model is evaluated using Root Mean Squared Error (RMSE):
```
RMSE: 0.876
```

## Customization
You can customize the number of recommendations by modifying the `top_n` parameter in `recommender.py`:
```python
recommend_movies(user_id, svd_matrix, movies_df, top_n=10)
```

## Datasets
The project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) which contains user ratings for various movies.

## Dependencies
```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
```

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please reach out to [niteshkotian3@gmail.com].

