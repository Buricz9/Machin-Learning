from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

def optimize_lightgbm_with_randomsearch(X_train, y_train):
    # Wykorzystanie parametrów zbliżonych do podanego param_grid
    param_dist = {
        'n_estimators': [20,50,70,100],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 70],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Inicjalizacja modelu LightGBM
    model = LGBMClassifier(random_state=42)

    # Inicjalizacja RandomizedSearchCV z podanymi dyskretnymi zestawami parametrów
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=50,    # można zmniejszyć, bo mamy mniej kombinacji
        cv=5, 
        n_jobs=-1, 
        scoring='roc_auc', 
        verbose=1, 
        random_state=42
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_model, best_params
