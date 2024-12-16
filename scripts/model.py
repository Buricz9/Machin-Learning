from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

def optimize_xgboost_with_random_search(X_train, y_train):
    # Definicja przestrzeni hiperparametrów zgodna z Twoim standardem
    param_dist = {
        'n_estimators': [20, 50, 70, 100],
        'max_depth': [1, 5, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # Inicjalizacja modelu XGBoost
    model = XGBClassifier(
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  # Liczba prób losowego przeszukiwania
        cv=5,
        n_jobs=-1,
        random_state=42,
        scoring='roc_auc'
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
