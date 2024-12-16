from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def train_svm_model_with_randomized_search(X_train, y_train):
    # Dyskretna przestrzeń przeszukiwania hiperparametrów
    param_dist = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],  # Dyskretne wartości C
        'gamma': [0.001, 0.01, 0.1, 1],  # Dyskretne wartości gamma
        'kernel': ['rbf', 'linear']  # Rodzaje jąder
    }

    # Inicjalizacja modelu SVM z class_weight='balanced'
    model = SVC(probability=True, class_weight='balanced', random_state=42)

    # RandomizedSearchCV dla optymalizacji hiperparametrów
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,  # Liczba iteracji losowego przeszukiwania
        cv=5,  # Walidacja krzyżowa
        scoring='roc_auc',  # Metryka do optymalizacji
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Trening modelu z RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Zwracanie najlepszego modelu i parametrów
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

    return best_model, best_params
