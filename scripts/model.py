from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_random_forest_model(X_train, y_train):
    # Inicjalizacja modelu Random Forest
    model = RandomForestClassifier(random_state=42)

    # Definiowanie siatki parametrów do przeszukiwania dla GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Inicjalizacja GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0, scoring='roc_auc')

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    return grid_search.best_estimator_, grid_search.best_params_
