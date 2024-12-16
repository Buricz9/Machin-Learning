import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

def train_random_forest_model_with_optuna(X_train, y_train):
    def objective(trial):
        param_grid = {
            'n_estimators': trial.suggest_categorical('n_estimators', [20, 50, 70, 100]),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 5, 10]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 4]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
        }

        model = RandomForestClassifier(
            n_estimators=param_grid['n_estimators'],
            max_depth=param_grid['max_depth'] if param_grid['max_depth'] != -1 else None,
            min_samples_split=param_grid['min_samples_split'],
            min_samples_leaf=param_grid['min_samples_leaf'],
            bootstrap=param_grid['bootstrap'],
            random_state=42
        )

        # Walidacja krzyżowa 5-krotna
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Zwracamy średnią wartość AUC z walidacji krzyżowej
        return scores.mean()

    # Tworzenie obiektu study z Optuny i optymalizacja
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Pobranie najlepszego zestawu hiperparametrów
    best_params = study.best_params

    # Tworzenie modelu na podstawie najlepszych parametrów i trening na całym zbiorze treningowym
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'] if best_params['max_depth'] != -1 else None,
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        bootstrap=best_params['bootstrap'],
        random_state=42
    )
    best_model.fit(X_train, y_train)

    return best_model, best_params
