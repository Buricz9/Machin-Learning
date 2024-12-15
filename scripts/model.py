from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def optimize_logistic_regression_with_random_search(X_train, y_train):
    # Definiowanie przestrzeni hiperparametrów dla RandomizedSearchCV
    param_dist = {
        'C': np.logspace(-4, 4, 20),  # Parametr regularyzacyjny C
        'solver': ['liblinear', 'lbfgs'],  # Różne algorytmy optymalizacji
        'penalty': ['l2']  # Tylko l2 dla 'lbfgs', 'liblinear' może obsługiwać 'l1'
    }

    # Inicjalizacja modelu Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        verbose=0,
        scoring='roc_auc',
        random_state=42
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Wyciąganie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
