from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def optimize_logistic_regression_with_randomsearch(X_train, y_train):
    # Definiowanie przestrzeni hiperparametrów
    param_dist = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Dyskretne wartości C
        'solver': ['liblinear', 'lbfgs'],  # Metody optymalizacji
        'penalty': ['l2']  # Regularizacja l2
    }

    # Inicjalizacja modelu Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Liczba iteracji – można dostosować
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        scoring='roc_auc'
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Wyciąganie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
