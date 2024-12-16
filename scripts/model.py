from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

def optimize_lightgbm_with_grid_search(X_train, y_train):
    # Definicja przestrzeni hiperparametrów dla GridSearchCV
    param_grid = {
        'n_estimators': [20,50,70,100],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 70],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Inicjalizacja modelu LightGBM
    model = LGBMClassifier(random_state=42)

    # Inicjalizacja GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params
