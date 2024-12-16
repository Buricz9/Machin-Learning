from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest_model(X_train, y_train):
    # Inicjalizacja modelu Random Forest
    model = RandomForestClassifier(random_state=42)

    # Definiowanie siatki parametrów do przeszukiwania dla GridSearchCV
    param_grid = {
        'n_estimators': [20, 50, 70, 100],
        'max_depth': [-1, 5, 10],  # -1 reprezentuje brak ograniczenia
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Inicjalizacja GridSearchCV
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

    # Zwrócenie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params
