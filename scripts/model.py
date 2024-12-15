from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def train_random_forest_model_with_random_search(X_train, y_train):
    # Definiowanie siatki parametrów do przeszukiwania dla RandomizedSearchCV
    param_dist = {
        'n_estimators': np.arange(50, 1001, 50),
        'max_depth': np.arange(10, 51, 5),
        'min_samples_split': np.arange(2, 21, 2),
        'min_samples_leaf': np.arange(1, 11, 1),
        'bootstrap': [True, False]
    }

    # Inicjalizacja modelu Random Forest
    model = RandomForestClassifier(random_state=42)

    # Inicjalizacja RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='roc_auc',
        random_state=42
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    return best_model, best_params
