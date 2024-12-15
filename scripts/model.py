from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def optimize_logistic_regression_with_grid_search(X_train, y_train):
    # Definiowanie przestrzeni hiperparametrów dla GridSearchCV
    param_grid = {
        'C': np.logspace(-4, 4, 20),  # Parametr regularyzacyjny C
        'solver': ['liblinear', 'lbfgs'],  # Różne algorytmy optymalizacji
        'penalty': ['l2']  # Tylko l2 dla 'lbfgs', 'liblinear' może obsługiwać 'l1'
    }

    # Inicjalizacja modelu Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # GridSearchCV dla optymalizacji hiperparametrów
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=0,
        scoring='roc_auc'
    )

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Wyciąganie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params
