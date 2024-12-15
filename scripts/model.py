import optuna
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_svm_model_with_optuna(X_train, y_train):
    # Definicja funkcji celu dla Optuna
    def objective(trial):
        # Proponowane wartości hiperparametrów
        C = trial.suggest_loguniform('C', 1e-4, 1e2)
        gamma = trial.suggest_loguniform('gamma', 1e-4, 1e1)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])

        # Inicjalizacja modelu SVM
        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)

        # Trening modelu
        model.fit(X_train, y_train)

        # Obliczanie AUC na zbiorze treningowym
        auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        return auc

    # Tworzenie badania Optuna i optymalizacja
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    # Zwracanie najlepszego modelu i parametrów
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model = SVC(**best_params, probability=True, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params
