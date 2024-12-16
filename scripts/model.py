import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_random_forest_model_with_optuna(X_train, y_train):
    # Funkcja celu dla Optuna
    def objective(trial):
        # Dyskretna przestrzeń hiperparametrów (zgodna z ustalonym wzorcem param_grid)
        param_grid = {
            'n_estimators': trial.suggest_categorical('n_estimators', [20, 50, 70, 100]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        # Inicjalizacja modelu Random Forest
        model = RandomForestClassifier(
            n_estimators=param_grid['n_estimators'],
            max_depth=param_grid['max_depth'] if param_grid['max_depth'] != -1 else None,
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            bootstrap=param_grid['bootstrap'],
            random_state=42
        )

        # Ocena modelu za pomocą AUC na zbiorze treningowym
        auc = roc_auc_score(y_train, model.fit(X_train, y_train).predict_proba(X_train)[:, 1])
        return auc

    # Tworzenie obiektu study i optymalizacja
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Pobranie najlepszego zestawu hiperparametrów
    best_params = study.best_params

    # Tworzenie modelu na podstawie najlepszych parametrów
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'] if best_params['max_depth'] != -1 else None,
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    # Trening najlepszego modelu
    best_model.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    return best_model, best_params
