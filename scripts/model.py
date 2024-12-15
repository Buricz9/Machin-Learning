from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def optimize_xgboost_with_grid_search(X_train, y_train):
    # Definiowanie przestrzeni hiperparametrów dla GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Inicjalizacja modelu XGBoost
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # GridSearchCV dla optymalizacji hiperparametrów
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

    # Wyciąganie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params
