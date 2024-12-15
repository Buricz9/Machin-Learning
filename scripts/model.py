import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def optimize_xgboost_with_optuna(X_train, y_train):
    def objective(trial):
        # Definicja przestrzeni hiperparametrów dla Optuna
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1)
        }

        # Inicjalizacja modelu XGBoost
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **params)

        # Trening modelu
        model.fit(X_train, y_train)

        # Obliczanie metryki AUC na zbiorze treningowym
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_pred_proba)

        return auc

    # Tworzenie studium optymalizacyjnego
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)

    # Wyciąganie najlepszego modelu i parametrów
    best_params = study.best_params
    best_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_params
