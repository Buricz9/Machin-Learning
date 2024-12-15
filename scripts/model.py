import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def train_random_forest_model_with_optuna(X_train, y_train):
    # Funkcja celu dla Optuna
    def objective(trial):
        # Definiowanie przestrzeni poszukiwania dla hiperparametrów
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 10, 50)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])

        # Inicjalizacja modelu Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            bootstrap=bootstrap,
            random_state=42
        )

        # Trening modelu
        model.fit(X_train, y_train)
        # Obliczenie metryki AUC na zbiorze treningowym
        auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        return auc

    # Tworzenie obiektu study i optymalizacja
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Pobranie najlepszego zestawu hiperparametrów i stworzenie modelu
    best_params = study.best_params
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    # Trening najlepszego modelu
    best_model.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    return best_model, best_params
