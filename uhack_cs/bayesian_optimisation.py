import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# Custom function to use weighted ensemble
def weighted_ensemble(X_train, y_train, X_test, y_test, weights):
    # Initialize the models
    lr_model = LogisticRegression()
    rf_model = RandomForestClassifier()
    svm_model = SVC(probability=True)
    gb_model = GradientBoostingClassifier()
    knn_model = KNeighborsClassifier()

    # Fit the models on the training data
    lr_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    knn_model.fit(X_train, y_train)

    # Make predictions (probabilities) for each model
    lr_preds = lr_model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
    rf_preds = rf_model.predict_proba(X_test)[:, 1]
    svm_preds = svm_model.predict_proba(X_test)[:, 1]
    gb_preds = gb_model.predict_proba(X_test)[:, 1]
    knn_preds = knn_model.predict_proba(X_test)[:, 1]

    # Combine predictions using the weighted sum
    final_preds = (weights['a'] * lr_preds + 
                   weights['b'] * rf_preds + 
                   weights['c'] * svm_preds + 
                   weights['d'] * gb_preds + 
                   weights['e'] * knn_preds)
    
    # Convert probabilities to binary predictions (0 or 1)
    final_preds_binary = (final_preds > 0.5).astype(int)
    
    # Calculate the F1 score
    f1 = f1_score(y_test, final_preds_binary)
    
    return f1

# Bayesian Optimization function
def bayesian_optimization(X_train, y_train, X_test, y_test):
    # Define the objective function
    def objective(a, b, c, d, e):
        # Ensure that the weights sum to 1
        if a + b + c + d + e != 1:
            return 0
        weights = {
            'a': a,
            'b': b,
            'c': c,
            'd': d,
            'e': e
        }
        f1 = weighted_ensemble(X_train, y_train, X_test, y_test, weights)
        return f1

    # Set up Bayesian optimizer
    pbounds = {
        'a': (0, 1),
        'b': (0, 1),
        'c': (0, 1),
        'd': (0, 1),
        'e': (0, 1),
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # Maximize the objective (F1 score)
    optimizer.maximize(init_points=10, n_iter=50)

    # Get the best parameters and score
    best_params = optimizer.max['params']
    best_f1 = optimizer.max['target']
    
    return best_params, best_f1

# Usage example
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # Example dataset (replace with your own dataset)
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run Bayesian Optimization to find optimal weights
    optimal_weights, optimal_f1 = bayesian_optimization(X_train, y_train, X_test, y_test)
    
    print(f"Optimal Weights: {optimal_weights}")
    print(f"Best F1 Score: {optimal_f1:.4f}")
