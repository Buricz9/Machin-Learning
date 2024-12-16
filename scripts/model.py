def train_svm_model_with_optuna(X_train, y_train):
    import optuna
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        # Dyskretna przestrzeń hiperparametrów
        C = trial.suggest_categorical('C', [0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        gamma = trial.suggest_categorical('gamma', [0.001, 0.01, 0.1, 1])
        kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])

        # Dodanie zbalansowania wag klas
        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, class_weight='balanced', random_state=42)
        
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        return auc

    # Optymalizacja Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1, timeout=600)

    best_params = study.best_params
    best_model = SVC(**best_params, probability=True, class_weight='balanced', random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, best_params
