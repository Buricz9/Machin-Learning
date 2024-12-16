from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest_model_with_random_search(X_train, y_train):
    # Dyskretny zestaw parametrów zgodny z wcześniejszym wzorcem
    param_dist = {
        'n_estimators': [20, 50, 70, 100],
        'max_depth': [-1, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Inicjalizacja modelu Random Forest
    model = RandomForestClassifier(random_state=42)

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Liczba iteracji dostosowana do liczby kombinacji w `param_dist`
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='roc_auc',
        random_state=42
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    return best_model, best_params
