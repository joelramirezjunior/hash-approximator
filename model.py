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
    Load and describe the data from a CSV or pickle file.

    Args:
        file_path (str): The path to the CSV or pickle file containing the data.

    Returns:
        pd.DataFrame: The data loaded from the file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        data = pd.read_pickle(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or pickle file.")
    
    print("Data Shape:", data.shape)
    print(data.dtypes)
    return data

def train_models(X_train, y_train):
    """
    Train various regression models on the training data.

    Args:
        X_train (pd.DataFrame): The training feature data.
        y_train (pd.Series): The training target data.

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
        X_test (pd.DataFrame): The testing feature data.
        y_test (pd.Series): The testing target data.
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
    data = load_and_describe_data("dataset.csv")  # Assuming CSV, change if it's pickle
    X = data[['seed1', 'seed2', 'seed3']]
    y = data['random_number']
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    models = train_models(X_train, y_train)
    scores_df = evaluate_models(models, X_test, y_test)
    
    print(scores_df)

if __name__ == "__main__":
    main()
