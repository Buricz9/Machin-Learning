# model.py
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np

def optimize_lightgbm_with_randomsearch(X_train, y_train):
    # Definicja przestrzeni hiperparametrów do losowego przeszukiwania
    param_dist = {
        'n_estimators': np.arange(100),
        'max_depth': np.arange(5, 30, 5),
        'learning_rate': np.linspace(0.01, 0.2, 5),
        'num_leaves': np.arange(20, 150, 20),
        'subsample': np.linspace(0.6, 1.0, 5),
        'colsample_bytree': np.linspace(0.6, 1.0, 5)
    }

    # Inicjalizacja modelu LightGBM
    model = LGBMClassifier(random_state=42)

    # Inicjalizacja RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, scoring='roc_auc', verbose=1, random_state=42)

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_

    return best_model, best_params
