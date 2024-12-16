from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

def optimize_logistic_regression_with_random_search(X_train, y_train):
    # Dyskretna przestrzeń hiperparametrów
    param_dist = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Dyskretne wartości C
        'solver': ['liblinear', 'lbfgs'],  # Różne algorytmy optymalizacji
        'penalty': ['l2']  # Tylko l2 dla lbfgs, liblinear obsługuje l1
    }

    # Inicjalizacja modelu Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Liczba iteracji – dostosowano dla szybszego działania
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
