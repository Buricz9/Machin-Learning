import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def optimize_lightgbm_with_optuna(X_train, y_train):
    # Funkcja celu do optymalizacji z użyciem Optuna
    def objective(trial):
        # Przestrzeń hiperparametrów
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100),
            'max_depth': trial.suggest_int('max_depth', -1, 50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        }

        # Inicjalizacja modelu LightGBM
        model = LGBMClassifier(**param, random_state=42)

        # Trening modelu
        model.fit(X_train, y_train)

        # Predykcja prawdopodobieństwa
        y_proba = model.predict_proba(X_train)[:, 1]
        
        # Obliczanie AUC
        auc = roc_auc_score(y_train, y_proba)
        return auc

    # Inicjalizacja optymalizacji Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Zwrócenie najlepszego modelu i parametrów
    best_params = study.best_trial.params
    best_model = LGBMClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params
