import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

def optimize_lightgbm_with_optuna(X_train, y_train):
    def objective(trial):
        # Parametry podobne do zdefiniowanych w param_grid dla GridSearch
        param = {
            'n_estimators': trial.suggest_categorical('n_estimators', [20, 50, 70, 100]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.1, 0.2]),
            'num_leaves': trial.suggest_categorical('num_leaves', [31, 50, 70]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0])
        }

        # Inicjalizacja modelu LightGBM z dobranymi parametrami
        model = LGBMClassifier(**param, random_state=42)

        # Trening modelu
        model.fit(X_train, y_train)

        # Predykcja prawdopodobieństwa na zbiorze treningowym
        y_proba = model.predict_proba(X_train)[:, 1]

        # Obliczanie AUC na zbiorze treningowym
        auc = roc_auc_score(y_train, y_proba)
        return auc

    # Inicjalizacja i uruchomienie optymalizacji za pomocą Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Zwrócenie najlepszego modelu i parametrów
    best_params = study.best_trial.params
    best_model = LGBMClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params
