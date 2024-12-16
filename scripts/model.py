from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def optimize_logistic_regression_with_grid_search(X_train, y_train):
    # Zmieniamy param_grid, aby użyć dyskretnych wartości C zamiast logspace
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    }

    model = LogisticRegression(random_state=42, max_iter=1000)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=0,
        scoring='roc_auc'
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params
