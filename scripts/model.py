from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def optimize_xgboost_with_grid_search(X_train, y_train):
    # Poprawiona siatka parametrów zgodnie z ustalonym wzorcem
    param_grid = {
        'n_estimators': [20, 50, 70, 100],       # Liczba drzew
        'max_depth': [-1, 5, 10],                # Maksymalna głębokość drzewa
        'learning_rate': [0.01, 0.1, 0.2],       # Współczynnik uczenia
        'num_leaves': [31, 50, 70],              # Liczba liści w drzewie
        'subsample': [0.6, 0.8, 1.0]             # Ułamek próbek do użycia w każdym drzewie
    }

    # Inicjalizacja modelu XGBoost
    model = XGBClassifier(
        random_state=42, 
        use_label_encoder=False,  # Usunięto ostrzeżenie
        eval_metric='logloss'
    )

    # Użycie StratifiedKFold dla walidacji krzyżowej
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # GridSearchCV dla optymalizacji hiperparametrów
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',  # Optymalizacja pod AUC
        n_jobs=-1,
        verbose=1
    )

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params
