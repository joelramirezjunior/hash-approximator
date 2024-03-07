import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def load_and_describe_data(file_path):
    """
    Load and describe the data from a numpy file.

    Args:
        file_path (str): The path to the .npy file containing the data.

    Returns:
        numpy.ndarray: The data loaded from the file.
    """
    data = np.load(file_path, allow_pickle=True)
    print("Data Shape:", data.shape)
    return data

def train_models(X_train, y_train):
    """
    Train various regression models on the training data.

    Args:
        X_train (numpy.ndarray): The training feature data.
        y_train (numpy.ndarray): The training target data.

    Returns:
        dict: A dictionary containing the trained models.
    """
    models = {
        "Linear Regression": LinearRegression().fit(X_train, y_train),
        "Decision Tree": DecisionTreeRegressor().fit(X_train, y_train),
        "SGD": SGDRegressor().fit(X_train, y_train),
        "Gradient Boosting": GradientBoostingRegressor().fit(X_train, y_train),
        "K-Nearest Neighbors": KNeighborsRegressor().fit(X_train, y_train)
    }
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate the trained models on the testing data.

    Args:
        models (dict): The trained models.
        X_test (numpy.ndarray): The testing feature data.
        y_test (numpy.ndarray): The testing target data.
    """
    model_scores = {'Model': [], 'R2 Score': [], 'MAE': [], 'RMSE': []}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(((y_test - y_pred) ** 2).mean())

        model_scores['Model'].append(name)
        model_scores['R2 Score'].append(r2)
        model_scores['MAE'].append(mae)
        model_scores['RMSE'].append(rmse)

    return pd.DataFrame(model_scores)

def main():
    data = load_and_describe_data("dataset.npy")
    X = data[:, 0:3]
    y = data[:, 3:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = train_models(X_train, y_train)
    scores_df = evaluate_models(models, X_test, y_test)
    
    print(scores_df)

if __name__ == "__main__":
    main()
