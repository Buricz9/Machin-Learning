from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm_model_with_grid_search(X_train, y_train):
    # Dyskretna przestrzeń hiperparametrów (zgodnie z ustalonym wzorcem)
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Dyskretne wartości C
        'gamma': [0.001, 0.01, 0.1, 1],               # Dyskretne wartości gamma
        'kernel': ['rbf', 'linear']                   # Typ kernela
    }

    # Inicjalizacja modelu SVM
    model = SVC(probability=True, random_state=42)

    # GridSearchCV dla optymalizacji hiperparametrów
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='roc_auc'
    )

    # Trening modelu z GridSearchCV
    grid_search.fit(X_train, y_train)

    # Zwrócenie najlepszego modelu i parametrów
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params
