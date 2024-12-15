from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

def train_logistic_regression_model_with_grid_search(X_train, y_train):
    # Definiowanie przestrzeni przeszukiwania hiperparametrów
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Różne wartości hiperparametru C
        'solver': ['liblinear', 'lbfgs'],  # Algorytmy optymalizacji
        'penalty': ['l1', 'l2']  # Rodzaje kar
    }

    # Inicjalizacja modelu Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)

    # GridSearchCV do optymalizacji hiperparametrów
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # Walidacja krzyżowa
        scoring='roc_auc',  # Metryka optymalizacji
        n_jobs=-1
    )

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Zwracanie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params
