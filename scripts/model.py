import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

def optimize_logistic_regression_with_optuna(X_train, y_train):
    # Funkcja celu dla Optuna
    def objective(trial):
        # Przestrzeń hiperparametrów
        C = trial.suggest_loguniform('C', 1e-4, 1e2)  # C z przedziału log-uniform
        solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
        penalty = 'l2' if solver == 'lbfgs' else trial.suggest_categorical('penalty', ['l1', 'l2'])
        
        # Inicjalizacja modelu Logistic Regression
        model = LogisticRegression(C=C, solver=solver, penalty=penalty, random_state=42, max_iter=1000)

        # Ocena modelu za pomocą walidacji krzyżowej
        auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return auc

    # Tworzenie i optymalizacja badania Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)

    # Wyciąganie najlepszych hiperparametrów
    best_params = study.best_params

    # Tworzenie najlepszego modelu na podstawie znalezionych parametrów
    best_model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
    best_model.fit(X_train, y_train)

    return best_model, best_params
