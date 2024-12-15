from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

def optimize_xgboost_with_random_search(X_train, y_train):
    # Definicja przestrzeni hiperparametrów dla RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
    }

    # Inicjalizacja modelu XGBoost
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

    # Inicjalizacja RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=5, n_jobs=-1, random_state=42, scoring='roc_auc')

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
